"""
Monte Carlo ABM baseline: location PDF and Phase 1 metrics per incident.

This is the reference surface for the whole project. It runs the existing
`lp_model.simulate` (unchanged) on one extracted incident and turns its
output into the canonical probability surface defined in `pdf_common`,
plus the per-incident numbers Phase 1 needs to lock down so the baseline
never has to be re-run.

Two surfaces are produced from one run, because they answer two different
questions and cost nothing extra:

  * pdf_endpoint  - KDE of where the replicates END UP (final_cells). This
    is "where is the subject likely to be found", the object the pending
    question was about ("the points ... are where the abm was found").
  * pdf_occupancy - KDE of where the replicates SPENT TIME (the ABM's own
    accumulated `density_frames[-1]`). This is probability-of-area over
    the whole search window, and it is the more natural comparand for the
    SNN's stationary belief, which is an equilibrium occupancy, not an
    endpoint snapshot.

Which of the two is the "canonical" baseline is a stated modelling choice
(see PRIMARY_SURFACE and the note in `run_incident`); both are saved so
the choice can be revisited without re-running anything.

Nothing here imports config, so it runs from the raw incident folders
alone. Paths and the incident list live in the batch runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lp_model
import pdf_common as pc

# --- Baseline run settings (stated, fixed for reproducibility) ---------

N_SAMPLES = 500          # replicates per incident, matching the paper
DURATION_HOURS = 24.0    # elapsed time the endpoint PDF is read at (see note)
BASE_SEED = 20260824     # fixed so the whole 65-incident set is reproducible

# Which surface downstream code treats as THE baseline location PDF.
# "occupancy" is the safer default: at long durations the endpoint surface
# piles mass on the map boundary (agents walk to the edge and stick), which
# is unphysical, whereas occupancy is robust to that. Set to "endpoint" if
# you specifically want the found-location snapshot and have chosen a
# duration short enough that boundary pile-up is negligible.
PRIMARY_SURFACE = "occupancy"

# --- Compute-cost accounting (Horowitz 2014, 45 nm CMOS) ---------------
#
# Hardware-agnostic operation count converted to an energy figure, so the
# ABM baseline has a number the SNN's synaptic-operation count can be put
# next to. This is a documented estimate and a LOWER BOUND: it counts
# compute only, not memory movement (see methodology 1.7). Both constants
# are deliberately explicit so they can be refined or swapped for your
# existing computational_cost() without touching the rest of the file.
FLOPS_PER_AGENT_STEP = 60      # ~ the float ops one agent's update costs
PJ_PER_FLOP = 1.0             # ~32-bit MAC at 45 nm, Horowitz order of magnitude


def load_incident(incident_dir):
    """
    Load one extracted incident: the Terrain (via lp_model), plus the IPP
    and found-location cells and grid metadata read from metadata.json /
    stack.npz. lp_model.Terrain does not surface the ground-truth cells,
    so they are read here.
    """
    incident_dir = Path(incident_dir)
    terrain = lp_model.Terrain.from_query_dir(incident_dir)

    meta = {}
    meta_path = incident_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    ipp_cell = tuple(meta.get("ipp_cell") or terrain.centre_cell())

    find_cell = meta.get("find_cell")
    find_in_bounds = meta.get("find_in_bounds")
    if find_cell is None or find_in_bounds is False:
        # Fall back to the stack sentinel if metadata is absent.
        with np.load(incident_dir / "stack.npz") as stack:
            if "find_cell" in stack.files:
                fc = stack["find_cell"]
                if fc[0] >= 0:
                    find_cell = (int(fc[0]), int(fc[1]))
    if find_cell is not None:
        find_cell = (int(find_cell[0]), int(find_cell[1]))
        h, w = terrain.shape
        in_b = (0 <= find_cell[0] < h) and (0 <= find_cell[1] < w)
        find_in_bounds = bool(in_b) if find_in_bounds is None else bool(find_in_bounds)
    else:
        find_in_bounds = False

    return {
        "dir": incident_dir,
        "terrain": terrain,
        "ipp_cell": (int(ipp_cell[0]), int(ipp_cell[1])),
        "find_cell": find_cell,
        "find_in_bounds": find_in_bounds,
        "index": meta.get("incident_index"),
    }


def run_incident(
    incident_dir,
    n_samples=N_SAMPLES,
    duration_hours=DURATION_HOURS,
    seed=BASE_SEED,
    pmf=lp_model.HIKER_PMF,
    pdf_cell_m=pc.PDF_CELL_M,
    bandwidth_m=pc.BANDWIDTH_M,
):
    """
    Run the ABM once and build both PDFs plus the Phase 1 metric row.

    Returns a dict: terrain/ground-truth info, the two PDFs and their grid
    info, the raw lp_model result, and a flat `metrics` dict ready to be a
    CSV row.
    """
    inc = load_incident(incident_dir)
    terrain = inc["terrain"]
    find_cell = inc["find_cell"] if inc["find_in_bounds"] else None

    result = lp_model.simulate(
        terrain,
        n_samples=n_samples,
        duration_hours=duration_hours,
        pmf=pmf,
        ipp_cell=inc["ipp_cell"],
        seed=seed,
        track_target_cell=find_cell,   # gives closest-approach for the energy statistic
    )

    # Canonical grid, computed once, and the water mask (lake / river
    # interiors) coarsened once and shared by both surfaces. The IPP and the
    # find location are known to be accessible (the simulation refuses an
    # inaccessible IPP, and the paper's inclusion rule excludes finds in
    # inaccessible areas), so they are exempted from the mask: otherwise a
    # shoreline find that lands in a majority-water 100 m PDF cell would be
    # zeroed, sending prob_at_find to 0 and search cost to the whole map.
    factor = pc.coarsen_factor(terrain.cell_size_m, pdf_cell_m)
    ph, pw = pc.grid_shape(terrain.shape, factor)
    water_pdf = pc.coarsen_mask(terrain.inaccessible, factor, (ph, pw))
    ipp_pdf = pc.cell_to_pdf_cell(inc["ipp_cell"], factor)
    find_pdf = pc.cell_to_pdf_cell(find_cell, factor) if find_cell else None
    water_pdf[ipp_pdf] = False
    if find_pdf is not None:
        water_pdf[find_pdf] = False

    # --- Endpoint surface: KDE of final_cells ---
    pdf_end, info = pc.points_to_pdf(
        result["final_cells"], terrain.shape, terrain.cell_size_m,
        pdf_cell_m=pdf_cell_m, bandwidth_m=bandwidth_m,
        inaccessible_pdf=water_pdf,
    )

    # --- Occupancy surface: KDE of the ABM's own accumulated density ---
    # density_frames are on lp_model's coarse grid (density_factor cells
    # each); expand the physical cell size accordingly before conforming.
    occ_field = result["density_frames"][-1]
    occ_cell_m = terrain.cell_size_m * result["density_factor"]
    pdf_occ, _ = pc.field_to_pdf(
        occ_field, occ_cell_m, terrain.shape, terrain.cell_size_m,
        pdf_cell_m=pdf_cell_m, bandwidth_m=bandwidth_m,
        inaccessible_pdf=water_pdf,
    )

    pdf_cell_size_m = info["pdf_cell_size_m"]
    primary = pdf_end if PRIMARY_SURFACE == "endpoint" else pdf_occ

    metrics = _metrics(
        result, primary, pdf_cell_size_m, ipp_pdf, find_pdf,
        inc, n_samples, duration_hours, seed,
    )
    # Diagnostic: how much endpoint mass sits on the outer ring of the map.
    metrics["endpoint_boundary_mass"] = float(_boundary_mass(pdf_end))

    return {
        "incident": inc,
        "result": result,
        "pdf_endpoint": pdf_end,
        "pdf_occupancy": pdf_occ,
        "pdf_primary_name": PRIMARY_SURFACE,
        "pdf_cell_size_m": pdf_cell_size_m,
        "factor": factor,
        "ipp_pdf": ipp_pdf,
        "find_pdf": find_pdf,
        "metrics": metrics,
    }


# --- Metrics -----------------------------------------------------------

def _metrics(result, pdf, pdf_cell_m, ipp_pdf, find_pdf, inc,
             n_samples, duration_hours, seed):
    stats = result["stats"]
    cell_km2 = (pdf_cell_m / 1000.0) ** 2

    m = {
        "incident_index": inc["index"],
        "find_in_bounds": inc["find_in_bounds"],
        "n_samples": n_samples,
        "duration_hours": duration_hours,
        "seed": seed,
        "cell_size_m": stats["cell_size_m"],
        "pdf_cell_size_m": pdf_cell_m,
        "total_steps": stats["total_steps"],
        # spread / confidence descriptors
        "entropy_bits": _entropy_bits(pdf),
        "n_modes": _count_modes(pdf),
        "peak_prob": float(pdf.max()),
        "mean_displacement_m": stats["mean_displacement_m"],
        "median_displacement_m": stats["median_displacement_m"],
        "p95_displacement_m": stats["p95_displacement_m"],
        "effective_speed_ms": stats["effective_speed_ms"],
        "area_covered_km2": stats["area_covered_km2"],
        "blocked_step_fraction": stats["blocked_step_fraction"],
        "layers_missing_n": len(stats["layers_missing"]),
    }

    # Ground-truth metrics: only meaningful when the true find location is
    # inside the extracted map.
    if find_pdf is not None:
        m["search_cost_km2"] = _search_cost_km2(pdf, find_pdf, cell_km2)
        m["prob_at_find"] = float(pdf[find_pdf])
        m["dist_peak_to_find_m"] = _dist_peak_to_find_m(pdf, find_pdf, pdf_cell_m)
        m["energy_statistic_m"] = _energy_statistic(
            result["closest_approach_cell"], inc["find_cell"], stats["cell_size_m"]
        )
    else:
        for k in ("search_cost_km2", "prob_at_find",
                  "dist_peak_to_find_m", "energy_statistic_m"):
            m[k] = None

    # Compute cost (Horowitz lower bound).
    agent_steps = int(stats["total_steps"]) * int(n_samples)
    op_count = agent_steps * FLOPS_PER_AGENT_STEP
    m["op_count"] = op_count
    m["est_energy_j"] = op_count * PJ_PER_FLOP * 1e-12
    return m


def _entropy_bits(pdf):
    p = pdf[pdf > 0]
    return float(-(p * np.log2(p)).sum())


def _count_modes(pdf, rel_thresh=0.10):
    """
    Rough count of distinct high-probability basins: local maxima whose
    height exceeds `rel_thresh` of the global peak. A planner reads this
    as "how many separate places worth searching".
    """
    from scipy.ndimage import maximum_filter, label
    peak = pdf.max()
    if peak <= 0:
        return 0
    strong = pdf >= rel_thresh * peak
    local_max = (pdf == maximum_filter(pdf, size=3)) & strong
    _, n = label(local_max)
    return int(n)


def _search_cost_km2(pdf, find_pdf, cell_km2):
    """
    SAR probability-of-area cost: rank every cell by assigned probability
    and read off how much ground (km2) must be searched before the cell
    containing the true location is reached.
    """
    thresh = pdf[find_pdf]
    n_cells = int(np.count_nonzero(pdf >= thresh))
    return float(n_cells * cell_km2)


def _dist_peak_to_find_m(pdf, find_pdf, pdf_cell_m):
    peak = np.unravel_index(int(np.argmax(pdf)), pdf.shape)
    dr = peak[0] - find_pdf[0]
    dc = peak[1] - find_pdf[1]
    return float(np.hypot(dr, dc) * pdf_cell_m)


def _energy_statistic(closest_cells, find_cell, cell_size_m):
    """
    Szekely-Rizzo energy statistic between each replicate's closest
    approach to the find location and the find location itself
    (Hashimoto et al. Eq. 2-3): 2*mean||x_i - y|| - mean||x_i - x_j||,
    in metres. Lower is better. Directly comparable to the paper.
    """
    if closest_cells is None:
        return None
    x = np.asarray(closest_cells, dtype=np.float64) * cell_size_m
    y = np.asarray(find_cell, dtype=np.float64) * cell_size_m
    cross = np.hypot(x[:, 0] - y[0], x[:, 1] - y[1]).mean()
    # Mean pairwise distance within the closest-approach set.
    diff = x[:, None, :] - x[None, :, :]
    within = np.hypot(diff[:, :, 0], diff[:, :, 1]).mean()
    return float(2.0 * cross - within)


def _boundary_mass(pdf, ring=1):
    """Fraction of probability sitting on the outer `ring` cells of the grid."""
    mask = np.zeros(pdf.shape, dtype=bool)
    mask[:ring, :] = mask[-ring:, :] = True
    mask[:, :ring] = mask[:, -ring:] = True
    return pdf[mask].sum()


# --- Heatmap rendering -------------------------------------------------

def render_heatmap(run, out_path, layer_style=None, cmap="magma",
                   surface=None, dpi=150, norm="log", log_decades=4.0,
                   show_axes=True):
    """
    Render a location-PDF heatmap over the incident's terrain, with a full
    terrain legend (only the layers actually present in the stack are
    listed) plus IPP (star) and found location (cross) markers. Uses
    sim_render to paint the same terrain background as the project's static
    maps, then overlays the chosen PDF, upsampled to full resolution, as a
    translucent colourmap.

    Parameters
    ----------
    surface : "endpoint" | "occupancy" | None
        Which PDF to draw; defaults to the primary one.
    norm : "log" | "linear"
        Colour-bar scaling. "log" (default) is the right choice for a
        lost-person surface: displacement from the IPP is long-tailed /
        approximately log-normal in the SAR literature, so a linear scale
        buries the tail structure where much of the search AREA lives. The
        log scaling is applied to the COLOUR MAPPING ONLY; the underlying
        PDF stays a true linear normalised probability, because Phase 2's
        KL and Wasserstein must run on the real probabilities.
    log_decades : float
        Dynamic range shown by the log colour bar, in orders of magnitude
        below the peak. 4 shows four decades; cells below that are treated
        as background and let the terrain show through.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm, Normalize

    import sim_render

    inc = run["incident"]
    terrain = inc["terrain"]
    pdf = run["pdf_endpoint"] if surface == "endpoint" else (
        run["pdf_occupancy"] if surface == "occupancy" else
        (run["pdf_endpoint"] if run["pdf_primary_name"] == "endpoint"
         else run["pdf_occupancy"])
    )
    factor = run["factor"]

    # Terrain background, painted from the persisted stack. composite_layers
    # returns legend_entries for exactly the layers it actually drew (a
    # missing or empty layer is skipped), so the legend lists only what is
    # really in this incident's stack.
    with np.load(Path(inc["dir"]) / "stack.npz") as stack:
        layers = {k: stack[k] for k in stack.files if stack[k].ndim == 2}
    gradient = layers.get("gradient_edges", np.zeros(terrain.shape, np.uint8))
    if layer_style is None:
        layer_style = DEFAULT_LAYER_STYLE
    bg, legend_entries = sim_render.composite_layers(layer_style, layers, gradient)

    # Upsample the PDF to full grid resolution for overlay.
    up = np.repeat(np.repeat(pdf, factor, axis=0), factor, axis=1)
    up = up[: terrain.shape[0], : terrain.shape[1]]

    vmax = float(up.max())
    if norm == "log":
        vmin = vmax / (10.0 ** log_decades)
        color_norm = LogNorm(vmin=vmin, vmax=vmax)
        masked = np.ma.masked_where(up < vmin, up)
        cbar_label = "probability mass per PDF cell (log scale)"
    else:
        color_norm = Normalize(vmin=0.0, vmax=vmax)
        masked = np.ma.masked_where(up <= vmax * 1e-3, up)
        cbar_label = "probability mass per PDF cell"

    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    ax.imshow(bg, interpolation="nearest")
    im = ax.imshow(masked, cmap=cmap, alpha=0.72, interpolation="nearest",
                   norm=color_norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # Markers first in the legend, then the terrain layers.
    ir, ic = inc["ipp_cell"]
    ax.scatter([ic], [ir], marker="*", s=280, facecolor="black",
               edgecolor="white", linewidths=1.3, zorder=10)
    handles = [mlines.Line2D([], [], marker="*", linestyle="None", markersize=15,
                             markerfacecolor="black", markeredgecolor="white",
                             label="IPP")]
    if inc["find_cell"] is not None and inc["find_in_bounds"]:
        fr, fc = inc["find_cell"]
        ax.scatter([fc], [fr], marker="X", s=200, facecolor="lime",
                   edgecolor="black", linewidths=1.3, zorder=11)
        handles.append(mlines.Line2D([], [], marker="X", linestyle="None",
                                     markersize=13, markerfacecolor="lime",
                                     markeredgecolor="black", label="Found location"))
    handles += [mpatches.Patch(color=np.array(c) / 255.0, label=lab)
                for lab, c in legend_entries]

    surf_name = surface or run["pdf_primary_name"]
    idx = inc["index"]
    ax.set_title(f"Incident {idx}  -  MC baseline location PDF ({surf_name})")

    h, w = terrain.shape
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)   # north-up: row 0 (north edge) at the top
    if show_axes:
        # Cell-index axes matching the benchmark paper (Fig. 5): longitude is
        # the column, latitude is the row. Matplotlib picks round tick
        # positions; because the raster is north-up, latitude cells are
        # counted from the SOUTH edge so the y-axis rises northward, which
        # the formatter does by mapping a row value to (h - 1 - row).
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        # Round latitude values (0, 500, 1000, ...) placed at their rows.
        # Latitude counts up from the south edge, so a value lat sits at
        # row (h - 1 - lat). This keeps the labels round rather than the
        # row positions.
        loc = MaxNLocator(nbins=6, integer=True, steps=[1, 2, 2.5, 5, 10])
        lat_vals = [t for t in loc.tick_values(0, h - 1) if 0 <= t <= h - 1]
        ax.set_yticks([(h - 1) - t for t in lat_vals])
        ax.set_yticklabels([str(int(round(t))) for t in lat_vals])
        ax.set_xlabel("Longitude (x cells)")
        ax.set_ylabel("Latitude (y cells)")
        ax.tick_params(labelsize=10)
        for name, sp in ax.spines.items():
            sp.set_visible(name in ("left", "bottom"))
        legend_anchor = (0.5, -0.13)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        legend_anchor = (0.5, -0.02)

    # Legend below the axis so it never covers the surface or fights the
    # colour bar, wrapped across a few columns for the busier stacks.
    ncol = min(4, max(2, (len(handles) + 3) // 4))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=legend_anchor,
              ncol=ncol, fontsize=9, framealpha=0.9)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


# Full terrain style, matching the project's static maps (batch_extract_65),
# so the heatmap background and its legend are identical to map_static /
# map_landcover and can cover every layer present in a stack. Layers not in
# a given incident's stack are simply skipped by composite_layers and so do
# not appear in that incident's legend. `key == "gradient"` is special-cased
# by composite_layers to use the gradient mask passed in.
BASE_LAYER_STYLE = [
    ("river_interior",  "River interior",      (0, 0, 255),     0, True),
    ("lake_interior",   "Lake interior",       (190, 175, 240), 0, True),
    ("gradient",        "Elevation gradients", (160, 160, 160), 1, False),
    ("streams",         "Streams",             (135, 206, 235), 1, False),
    ("ditches",         None,                  (135, 206, 235), 1, False),
    ("river_banks",     "Riverbanks",          (255, 140, 0),   1, False),
    ("lake_shorelines", "Lake shorelines",     (255, 200, 0),   1, False),
    ("trails",          "Hiking trails",       (0, 0, 0),       1, False),
    ("railroads",       "Railroads",           (85, 170, 60),   1, False),
    ("powerlines",      "Powerline easements", (148, 0, 211),   1, False),
    ("roads",           "Roads",               (139, 0, 0),     1, False),
]

LANDCOVER_FILLS = [
    ("field",     "Field",     (238, 240, 205), 0, True),
    ("woodland",  "Woodland",  (198, 224, 180), 0, True),
    ("beach",     "Beach",     (245, 222, 179), 0, True),
    ("buildings", "Buildings", (168, 160, 152), 0, True),
]

DEFAULT_LAYER_STYLE = (
    LANDCOVER_FILLS
    + BASE_LAYER_STYLE
    + [("barriers", "Barriers / hedgerows", (120, 90, 40), 1, False)]
)