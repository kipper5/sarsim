"""
Agent-based lost person model for SARsim.

Implements the model of Hashimoto, Heintzman, Koester & Abaid (2022),
"An agent-based model reveals lost person behavior based on data from
wilderness search and rescue", Scientific Reports 12:5873.

The six reorientation strategies (RW, RT, DT, SP, VE, BT) are exactly as
defined in that paper's "Lost person behavior strategies" section, the
position update is their Eq. (1), and the default behaviour distribution
is the fitted hiker profile from their Eq. (5) / Fig. 7.

Two things differ deliberately from the reference MATLAB implementation,
both because SARsim grids are not fixed at 6.67 m/cell:

  * The time step is derived from the achieved cell size rather than
    hardcoded at 850 steps/hour. One step is the time to cross one cell
    at the maximum walking speed, which is what produces 850 steps/hour
    at the paper's 6.67 m cell -- so the reference number falls out of
    the formula rather than being asserted.
  * Agent state is carried as float and rounded only when the grid is
    indexed. Eq. (1) is a second-order filter; rounding its output back
    onto the lattice every step destroys the momentum it exists to
    create, and the effect gets worse as cells get larger.

Note on the second point: the paper's own MATLAB implementation appears
to round position to the lattice every step (it describes reorienting a
3x3 neighbourhood on the current cell via "appropriately rounding the
coordinate values"), so this is a genuine, disclosed divergence from a
strict reading of the reference implementation, not merely a more
general reformulation of an identical rule. Trajectories here carry more
directional persistence and less lattice-locked jitter than the paper's
own code would produce.

Everything is vectorised across Monte Carlo replicates: one step advances
all N agents with a fixed number of array operations, so cost scales with
step count rather than with N.
"""

import json
from pathlib import Path

import numpy as np

# --- Model constants (paper values) ------------------------------------

BEHAVIOURS = ("RW", "RT", "DT", "SP", "VE", "BT")
BEHAVIOUR_LABELS = {
    "RW": "Random walking",
    "RT": "Route traveling",
    "DT": "Direction traveling",
    "SP": "Staying put",
    "VE": "View enhancing",
    "BT": "Backtracking",
}
RW, RT, DT, SP, VE, BT = range(6)

# Fitted hiker profile, Fig. 7: [RW, RT, DT, SP, VE, BT].
HIKER_PMF = (0.055, 0.377, 0.559, 0.003, 0.006, 0.0)

# Table 1. Alpha is chosen in the paper to avoid the coefficient
# cancellation that a fair weighting of 0.5 produces in Eq. (1).
ALPHA = 0.55

# Maximum walking speed used to set the time step (m/s).
MAX_WALK_SPEED_MS = 1.575

# --- Terrain semantics -------------------------------------------------
#
# Which persisted stack layers play which role in the model. The paper's
# linear features are hiking trails, roads, railroads, powerline
# easements, water features and the elevation-gradient edges; its
# inaccessible areas are lake and wider-river interiors.
#
# Ditches are listed because SARsim splits them out of streams for KSTAT
# scoring; in the model they are simply another watercourse to follow.
# Barriers (hedgerows, walls, fence lines) are deliberately NOT linear
# features: a hedge is a thing you walk along the side of, not a route,
# and treating it as one would inflate route traveling in exactly the
# enclosed farmland where the model is already weakest.

LINEAR_FEATURE_KEYS = (
    "gradient_edges",
    "trails",
    "roads",
    "railroads",
    "powerlines",
    "streams",
    "ditches",
    "river_banks",
    "lake_shorelines",
)

INACCESSIBLE_KEYS = ("river_interior", "lake_interior")

# Buildings are excluded by default: a building footprint is a place
# people are found (it is its own KSTAT find category), not a wall.
OPTIONAL_INACCESSIBLE_KEYS = ("buildings",)

# Guard rails. A run that breaches these is almost always a typo in the
# duration box rather than an intention.
MAX_SAMPLES = 5000
MAX_STEPS = 250_000

# Backtracking history is a ring buffer this many positions deep. An
# agent that backtracks continuously for longer than this runs out of
# trail and stays put instead. With the hiker profile BT has probability
# zero, so this only binds on hand-edited profiles.
BACKTRACK_HISTORY = 1024

# Eight compass offsets as (drow, dcol), ordered by angle so that
# index k corresponds to k * 45 degrees. Body coordinates are just an
# index rotation on this table: forward is k, forward-left k-1,
# forward-right k+1.
COMPASS = np.array(
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
    dtype=np.int64,
)

# Nine offsets including staying in the current cell, for random walking.
MOORE9 = np.array(
    [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], dtype=np.int64
)

# Occupancy is accumulated onto a coarser grid than the terrain: roughly
# this many bins across the longer axis. Binning matters because what an
# operator needs from the animation is where the replicates concentrate,
# and single-cell counts on a 3000-cell grid are too sparse to read as a
# probability of area.
DENSITY_TARGET_BINS = 250


def normalise_pmf(pmf):
    """
    Coerce a six-element behaviour distribution to a valid PMF.

    Hand-entered profiles rarely sum to exactly one, and the paper's own
    hiker profile sums to 0.9999. Renormalising is the sane response;
    rejecting the input is not.
    """
    p = np.asarray(pmf, dtype=np.float64).ravel()
    if p.size != 6:
        raise ValueError(f"Behaviour PMF must have 6 entries, got {p.size}")
    if np.any(p < 0):
        raise ValueError("Behaviour PMF entries must be non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("Behaviour PMF must have at least one positive entry")
    return p / total


def steps_per_hour(cell_size_m, speed_ms=MAX_WALK_SPEED_MS):
    """
    Time steps per hour for a given cell size.

    One step is one cell traversed at `speed_ms`. At the paper's 6.67 m
    cell this returns 850, matching Table 1.
    """
    return 3600.0 * float(speed_ms) / float(cell_size_m)


class Terrain:
    """
    The model's view of one saved query: what can be followed, what
    cannot be entered, and how high the ground is.

    Built from the persisted stack rather than recomputed, so the
    simulation runs on exactly the layers the static map was drawn from.
    `layers_used` and `layers_missing` record what was actually found,
    which is what the UI reports back to the operator -- an empty linear
    layer changes route traveling into a random walk without raising
    anything, so it has to be visible.
    """

    def __init__(self, linear, inaccessible, elevation, cell_size_m,
                 layers_used, layers_missing, crs=None):
        self.linear = linear
        self.inaccessible = inaccessible
        self.elevation = elevation
        self.cell_size_m = float(cell_size_m)
        self.layers_used = layers_used
        self.layers_missing = layers_missing
        self.crs = crs
        self.shape = linear.shape

    @property
    def has_elevation(self):
        return self.elevation is not None

    @classmethod
    def from_query_dir(cls, query_dir, block_buildings=False):
        """
        Load the raster stack and metadata written by the extraction
        pipeline. Raises FileNotFoundError if the query predates stack
        persistence, which is a clearer failure than silently simulating
        on an empty map.
        """
        query_dir = Path(query_dir)
        stack_path = query_dir / "stack.npz"
        meta_path = query_dir / "metadata.json"

        if not stack_path.exists():
            raise FileNotFoundError(
                f"No raster stack at {stack_path}. Re-run the extraction for "
                "this query: simulation reads the persisted layers, not the "
                "rendered PNG."
            )

        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
        grid = meta.get("grid", {})

        with np.load(stack_path) as stack:
            available = set(stack.files)

            shape = None
            for key in stack.files:
                if stack[key].ndim == 2:
                    shape = stack[key].shape
                    break
            if shape is None:
                raise ValueError(f"No 2D layers found in {stack_path}")

            used, missing = [], []

            linear = np.zeros(shape, dtype=bool)
            for key in LINEAR_FEATURE_KEYS:
                if key in available:
                    layer = stack[key]
                    if np.count_nonzero(layer):
                        linear |= layer.astype(bool)
                        used.append(key)
                    else:
                        missing.append(f"{key} (present but empty)")
                else:
                    missing.append(key)

            blocked_keys = list(INACCESSIBLE_KEYS)
            if block_buildings:
                blocked_keys += list(OPTIONAL_INACCESSIBLE_KEYS)

            inaccessible = np.zeros(shape, dtype=bool)
            for key in blocked_keys:
                if key in available:
                    layer = stack[key]
                    if np.count_nonzero(layer):
                        inaccessible |= layer.astype(bool)
                        used.append(key)
                    else:
                        missing.append(f"{key} (present but empty)")
                else:
                    missing.append(key)

            elevation = None
            if "elevation" in available:
                elevation = stack["elevation"].astype(np.float32)
                used.append("elevation")
            else:
                missing.append("elevation")

        cell_size_m = grid.get("cell_size_m")
        if not cell_size_m:
            raise ValueError(
                "metadata.json has no grid.cell_size_m. Every distance and "
                "the time step itself are derived from it; guessing would "
                "silently rescale the results."
            )

        return cls(
            linear=linear,
            inaccessible=inaccessible,
            elevation=elevation,
            cell_size_m=cell_size_m,
            layers_used=used,
            layers_missing=missing,
            crs=grid.get("crs"),
        )

    def centre_cell(self):
        """
        The IPP. The extraction builds the grid square about the query
        centre, so the centre cell is the initial planning point -- the
        same convention as the paper, which places the IPP at the middle
        of its 3000 x 3000 grid.
        """
        return (self.shape[0] // 2, self.shape[1] // 2)


def _heading_index(velocity, rng):
    """
    Body-coordinate orientation: the compass direction nearest the
    current velocity.

    An agent with no velocity (it has just stayed put from rest) has no
    body frame, so one is drawn at random rather than defaulting to a
    fixed axis, which would bias direction traveling due east.
    """
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    angle = np.arctan2(velocity[:, 0], velocity[:, 1])
    k = np.rint(angle / (np.pi / 4.0)).astype(np.int64) % 8
    stalled = speed < 1e-9
    if np.any(stalled):
        k[stalled] = rng.integers(0, 8, size=int(np.count_nonzero(stalled)))
    return k


def _gather(mask_flat, rows, cols, shape, default=False):
    """Look up a flattened 2D array, returning `default` out of bounds."""
    h, w = shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(rows.shape, default, dtype=mask_flat.dtype)
    if np.any(inside):
        out[inside] = mask_flat[rows[inside] * w + cols[inside]]
    return out


def simulate(
    terrain,
    n_samples=500,
    duration_hours=12.0,
    pmf=HIKER_PMF,
    ipp_cell=None,
    alpha=ALPHA,
    seed=None,
    n_frames=80,
    speed_ms=MAX_WALK_SPEED_MS,
    block_at_boundary=True,
    track_target_cell=None,
    behaviour_selector=None,
):
    """
    Run `n_samples` Monte Carlo replicates of the lost person model.

    Parameters
    ----------
    terrain : Terrain
    n_samples : int
        Replicates. The paper uses 500 per LPT per incident.
    duration_hours : float
        Simulated elapsed time. The paper runs 100 h on a 20 km map so
        the agent can in principle cross it; scale this to the extent.
    pmf : sequence of 6 floats
        Behaviour distribution [RW, RT, DT, SP, VE, BT]. Defaults to the
        fitted hiker profile.
    ipp_cell : (row, col) or None
        Initial planning point. Defaults to the grid centre.
    n_frames : int
        Number of trajectory snapshots retained for animation.
    track_target_cell : (row, col) or None
        If given, each replicate's closest approach to this cell is
        tracked across the whole trajectory (not just its final
        position) and returned as "closest_approach_m" /
        "closest_approach_cell". This is what Hashimoto et al.'s own
        energy-statistic validation needs -- the real find location is
        the natural target when scoring a run against ground truth.
        Adds a negligible O(n_samples) cost per step; leave as None
        (default) for zero overhead when no ground truth is available.
    behaviour_selector : callable or None
        Optional override for the per-step categorical draw. Called
        once per step as behaviour_selector(n_samples, pmf, rng) and
        must return an (n_samples,) int array of behaviour indices in
        [0, 5]. Leave None for the baseline's own draw. This is the
        hook Architecture 1 uses to substitute a spiking race for this
        one mechanism, while every other line in the loop -- movement,
        terrain interaction, backtracking -- is reused unchanged.

    Returns
    -------
    dict with keys:
        frames              list of (n_samples, 2) int arrays of cell
                            positions
        frame_times_h       elapsed hours for each frame
        density_frames      cumulative occupancy counts on a coarse
                            grid, one per frame -- the probability of
                            area as it evolves
        density_factor      terrain cells per density bin
        final_cells         (n_samples, 2) int array
        ipp_cell            (row, col) actually used
        stats               dict of run diagnostics
        closest_approach_m  (n_samples,) float array of each replicate's
                            minimum distance in metres to
                            track_target_cell, or None if not tracked
        closest_approach_cell  (n_samples, 2) array of the cell each
                            replicate was at when it achieved that
                            minimum, or None if not tracked
    """
    pmf = normalise_pmf(pmf)
    n = int(n_samples)
    if n < 1:
        raise ValueError("Number of samples must be at least 1")
    if n > MAX_SAMPLES:
        raise ValueError(f"Number of samples is capped at {MAX_SAMPLES}")

    h, w = terrain.shape
    sph = steps_per_hour(terrain.cell_size_m, speed_ms)
    total_steps = int(round(float(duration_hours) * sph))
    if total_steps < 1:
        raise ValueError("Duration is too short to produce a single time step")
    if total_steps > MAX_STEPS:
        raise ValueError(
            f"{total_steps:,} time steps requested (cap {MAX_STEPS:,}). "
            f"At {terrain.cell_size_m:.1f} m/cell one hour is {sph:.0f} steps, "
            f"so reduce the duration or re-extract at a coarser resolution."
        )

    rng = np.random.default_rng(seed)

    if ipp_cell is None:
        ipp_cell = terrain.centre_cell()
    ipp = np.array(ipp_cell, dtype=np.float64)
    if terrain.inaccessible[int(ipp[0]), int(ipp[1])]:
        raise ValueError(
            "The IPP falls in an inaccessible cell (lake or river interior). "
            "No trajectory can leave it."
        )

    # Optional: track each replicate's closest approach to a target cell
    # (e.g. the real find location) across the whole trajectory, for
    # Hashimoto et al.'s own energy-statistic validation approach --
    # not just the final position, which is all `final_cells` gives you.
    track_target = None
    closest_dist = None
    closest_pos = None
    if track_target_cell is not None:
        track_target = np.array(track_target_cell, dtype=np.float64)
        closest_dist = np.full(n, np.inf, dtype=np.float64)
        closest_pos = np.zeros((n, 2), dtype=np.float64)

    linear_flat = terrain.linear.ravel()
    blocked_flat = terrain.inaccessible.ravel()
    elev_flat = (
        terrain.elevation.ravel() if terrain.has_elevation else None
    )

    # x(1) is the IPP; x(2) is drawn from the eight adjacent cells, which
    # is what gives the first step a velocity to orient body coordinates.
    prev = np.repeat(ipp[None, :], n, axis=0)
    pos = prev + COMPASS[rng.integers(0, 8, size=n)].astype(np.float64)
    np.clip(pos[:, 0], 0, h - 1, out=pos[:, 0])
    np.clip(pos[:, 1], 0, w - 1, out=pos[:, 1])

    # Backtracking trail: ring buffer of positions reached under a
    # non-BT behaviour, plus how far back along it each agent has walked.
    hist = np.zeros((n, BACKTRACK_HISTORY, 2), dtype=np.float32)
    hist[:, 0] = pos
    hist_write = np.ones(n, dtype=np.int64)
    hist_count = np.ones(n, dtype=np.int64)
    back_off = np.zeros(n, dtype=np.int64)
    prev_beh = np.full(n, RW, dtype=np.int64)

    cumulative_pmf = np.cumsum(pmf)
    behaviour_counts = np.zeros(6, dtype=np.int64)
    blocked_steps = 0

    # Frame schedule: snapshots evenly spaced in time.
    n_frames = max(2, min(int(n_frames), total_steps))
    frame_at = np.unique(
        np.linspace(0, total_steps - 1, n_frames).round().astype(np.int64)
    )
    frame_lookup = {int(s): i for i, s in enumerate(frame_at)}

    # Cumulative occupancy on the coarse grid. Counts accumulate every
    # step, not only at snapshots, so the density reflects the whole path
    # rather than 80 sampled instants of it.
    density_factor = max(1, int(np.ceil(max(h, w) / float(DENSITY_TARGET_BINS))))
    dh = int(np.ceil(h / density_factor))
    dw = int(np.ceil(w / density_factor))
    density = np.zeros(dh * dw, dtype=np.int64)

    frames = []
    frame_times = []
    density_frames = []

    for step in range(total_steps):
        cell = np.rint(pos).astype(np.int64)
        np.clip(cell[:, 0], 0, h - 1, out=cell[:, 0])
        np.clip(cell[:, 1], 0, w - 1, out=cell[:, 1])

        if track_target is not None:
            d = (np.hypot(cell[:, 0] - track_target[0], cell[:, 1] - track_target[1])
                 * terrain.cell_size_m)
            better = d < closest_dist
            if np.any(better):
                closest_dist[better] = d[better]
                closest_pos[better] = cell[better]

        density += np.bincount(
            (cell[:, 0] // density_factor) * dw + (cell[:, 1] // density_factor),
            minlength=dh * dw,
        )

        # Independent realisation of the behaviour PMF for every agent.
        # behaviour_selector, when supplied, replaces this categorical draw
        # with an alternative decision mechanism -- e.g. Architecture 1's
        # spiking race -- while every other line in this loop (movement,
        # terrain interaction, backtracking) is reused unchanged.
        if behaviour_selector is not None:
            beh = np.asarray(behaviour_selector(n, pmf, rng), dtype=np.int64)
            if beh.shape != (n,):
                raise ValueError(
                    f"behaviour_selector must return shape ({n},), got {beh.shape}"
                )
            np.clip(beh, 0, 5, out=beh)
        else:
            beh = np.searchsorted(cumulative_pmf, rng.random(n)).astype(np.int64)
            np.clip(beh, 0, 5, out=beh)
        behaviour_counts += np.bincount(beh, minlength=6)

        velocity = pos - prev
        k = _heading_index(velocity, rng)

        # Provisional update x_hat, in absolute cell coordinates.
        target = cell.astype(np.float64).copy()

        # --- Random walking: uniform over the 9 cells of the Moore
        # neighbourhood including the current one.
        m_rw = beh == RW
        n_rw = int(np.count_nonzero(m_rw))
        if n_rw:
            target[m_rw] = cell[m_rw] + MOORE9[rng.integers(0, 9, size=n_rw)]

        # --- Route traveling: uniform over whichever of the three
        # forward cells in body coordinates carry a linear feature;
        # a random walk if none of them do.
        m_rt = beh == RT
        n_rt = int(np.count_nonzero(m_rt))
        if n_rt:
            kk = k[m_rt]
            base = cell[m_rt]
            # (3, n_rt, 2): forward-left, forward, forward-right.
            opts = np.stack(
                [COMPASS[(kk - 1) % 8], COMPASS[kk], COMPASS[(kk + 1) % 8]]
            )
            cand = base[None, :, :] + opts
            has_feat = _gather(
                linear_flat, cand[:, :, 0], cand[:, :, 1], (h, w), default=False
            )
            count = has_feat.sum(axis=0)
            on_route = count > 0

            chosen = np.zeros((n_rt, 2), dtype=np.int64)
            if np.any(on_route):
                # Uniform pick among the available forward cells: walk the
                # cumulative count until it passes a scaled uniform draw.
                draw = rng.random(n_rt) * np.maximum(count, 1)
                cum = np.cumsum(has_feat, axis=0)
                pick = np.argmax((cum > draw[None, :]) & has_feat, axis=0)
                chosen[on_route] = cand[
                    pick[on_route], np.nonzero(on_route)[0], :
                ]
            if np.any(~on_route):
                n_off = int(np.count_nonzero(~on_route))
                chosen[~on_route] = (
                    base[~on_route] + MOORE9[rng.integers(0, 9, size=n_off)]
                )
            target[m_rt] = chosen

        # --- Direction traveling: one cell forward in body coordinates.
        m_dt = beh == DT
        if np.any(m_dt):
            target[m_dt] = cell[m_dt] + COMPASS[k[m_dt]]

        # --- Staying put: target already equals the current cell.

        # --- View enhancing: the highest of the eight neighbours, or
        # stay put if the agent is already on the local high point.
        m_ve = beh == VE
        n_ve = int(np.count_nonzero(m_ve))
        if n_ve and elev_flat is not None:
            base = cell[m_ve]
            cand = base[:, None, :] + COMPASS[None, :, :]
            neigh = _gather(
                elev_flat, cand[:, :, 0], cand[:, :, 1], (h, w),
                default=np.float32(-np.inf),
            )
            own = elev_flat[base[:, 0] * w + base[:, 1]]
            best = np.argmax(neigh, axis=1)
            rows = np.arange(n_ve)
            uphill = neigh[rows, best] > own
            ve_target = base.astype(np.float64)
            ve_target[uphill] = cand[rows[uphill], best[uphill], :]
            target[m_ve] = ve_target

        # --- Backtracking: retrace the recorded non-BT trail. A first BT
        # step returns to the previous position; consecutive BT steps walk
        # further back along it.
        m_bt = beh == BT
        n_bt = int(np.count_nonzero(m_bt))
        if n_bt:
            continuing = m_bt & (prev_beh == BT)
            back_off[m_bt & ~continuing] = 1
            back_off[continuing] += 1

            idx = np.nonzero(m_bt)[0]
            depth = back_off[idx]
            # hist[hist_write - 1] is always the agent's own current
            # position -- it was written there by whatever non-BT move
            # just happened. depth must skip past that self-entry, so a
            # depth of 1 (the first BT step of a streak) reads the entry
            # one further back, which is the actual previous location.
            available = (depth + 1) <= hist_count[idx]
            read = (hist_write[idx] - depth - 1) % BACKTRACK_HISTORY
            bt_target = cell[idx].astype(np.float64)
            if np.any(available):
                sel = idx[available]
                bt_target[available] = hist[sel, read[available]]
            target[idx] = bt_target
        back_off[~m_bt] = 0

        # --- Eq. (1): one step of memory smooths the trajectory to
        # something a walking person could plausibly have produced.
        x_hat = target
        v_prop = x_hat - pos
        new_pos = (2.0 - alpha) * pos + (alpha - 1.0) * prev + alpha * v_prop

        # Inaccessibility is checked on the smoothed position, not on the
        # provisional one: the smoothing can overshoot into water that
        # x_hat itself avoided. A blocked agent stays put for this step
        # but keeps its previous position, so the body frame survives and
        # it does not lose its heading against a shoreline.
        new_cell = np.rint(new_pos).astype(np.int64)
        inside = (
            (new_cell[:, 0] >= 0) & (new_cell[:, 0] < h)
            & (new_cell[:, 1] >= 0) & (new_cell[:, 1] < w)
        )
        ok = inside.copy()
        if np.any(inside):
            idx_in = np.nonzero(inside)[0]
            ok[idx_in] = ~blocked_flat[
                new_cell[idx_in, 0] * w + new_cell[idx_in, 1]
            ]
        if not block_at_boundary:
            ok |= ~inside

        blocked_steps += int(n - np.count_nonzero(ok))

        prev_next = np.where(ok[:, None], pos, prev)
        pos = np.where(ok[:, None], new_pos, pos)
        np.clip(pos[:, 0], 0, h - 1, out=pos[:, 0])
        np.clip(pos[:, 1], 0, w - 1, out=pos[:, 1])
        prev = prev_next

        # Extend the backtracking trail for every agent that moved under
        # a non-BT behaviour.
        movers = ~m_bt
        if np.any(movers):
            mv = np.nonzero(movers)[0]
            hist[mv, hist_write[mv] % BACKTRACK_HISTORY] = pos[mv]
            hist_write[mv] = (hist_write[mv] + 1) % BACKTRACK_HISTORY
            hist_count[mv] = np.minimum(hist_count[mv] + 1, BACKTRACK_HISTORY)

        prev_beh = beh

        if step in frame_lookup:
            snap = np.rint(pos).astype(np.int64)
            np.clip(snap[:, 0], 0, h - 1, out=snap[:, 0])
            np.clip(snap[:, 1], 0, w - 1, out=snap[:, 1])
            frames.append(snap)
            frame_times.append((step + 1) / sph)
            density_frames.append(density.reshape(dh, dw).astype(np.float32))

    final_cells = np.rint(pos).astype(np.int64)
    displacement_m = (
        np.hypot(final_cells[:, 0] - ipp[0], final_cells[:, 1] - ipp[1])
        * terrain.cell_size_m
    )

    stats = {
        "n_samples": n,
        "duration_hours": float(duration_hours),
        "total_steps": total_steps,
        "steps_per_hour": sph,
        "cell_size_m": terrain.cell_size_m,
        "seed": seed,
        "pmf": {b: float(p) for b, p in zip(BEHAVIOURS, pmf)},
        "behaviour_realisations": {
            b: int(c) for b, c in zip(BEHAVIOURS, behaviour_counts)
        },
        "blocked_step_fraction": blocked_steps / float(n * total_steps),
        "mean_displacement_m": float(displacement_m.mean()),
        "median_displacement_m": float(np.median(displacement_m)),
        "max_displacement_m": float(displacement_m.max()),
        "p95_displacement_m": float(np.percentile(displacement_m, 95)),
        # Effective speed as defined in the paper's Fig. 8 discussion:
        # net displacement per elapsed hour, an order of magnitude below
        # walking speed because the trajectories are convoluted. The
        # paper reports 0.064 m/s; a wildly different value here means
        # the smoothing or the time step has been mis-specified. Note
        # this is a different statistic to the paper's own Fig. 8 number
        # (which regresses time-to-closest-approach against real
        # incidents' IPP-find distances) -- it targets the same
        # headline figure as a rough plausibility check, not a
        # reproduction of that derivation.
        "effective_speed_ms": float(
            displacement_m.mean() / (float(duration_hours) * 3600.0)
        ),
        "area_covered_km2": float(
            np.count_nonzero(density_frames[-1])
            * (density_factor * terrain.cell_size_m) ** 2 / 1e6
        ),
        "linear_layers_used": list(terrain.layers_used),
        "layers_missing": list(terrain.layers_missing),
        "elevation_available": bool(terrain.has_elevation),
    }

    return {
        "frames": frames,
        "frame_times_h": frame_times,
        "density_frames": density_frames,
        "density_factor": density_factor,
        "final_cells": final_cells,
        "ipp_cell": (int(ipp[0]), int(ipp[1])),
        "stats": stats,
        "closest_approach_m": closest_dist,
        "closest_approach_cell": closest_pos,
    }