"""
Shared probability-surface representation for SARsim.

This module is deliberately model-agnostic. It defines the ONE canonical
grid, smoothing kernel and normalisation that every model in the project
must use when it hands back a location probability surface: the Monte
Carlo ABM baseline (Phase 1), and later the spiking MRF sampler
(Architecture 2). If two surfaces are built through different code here,
any KL or Wasserstein number computed between them in Phase 2 is partly
an artefact of the post-processing rather than a real difference in the
physics the two models captured. So both models call the same two
functions: `points_to_pdf` (for a cloud of continuous sample points, e.g.
ABM endpoints) and `field_to_pdf` (for an already-gridded count field,
e.g. per-neuron spike counts or the ABM occupancy grid).

Design decisions, all stated so they can go in the write-up:

  * Canonical PDF cell size is fixed in METRES (PDF_CELL_M), not in cells,
    so every incident lands on the same physical resolution regardless of
    the small per-incident differences in the extraction cell size. The
    IPP stays at the grid centre, matching the extraction convention.
  * Smoothing is a Gaussian of fixed physical width (BANDWIDTH_M), applied
    identically to both surfaces. This is "phi_ker": histogram, then blur,
    then normalise. It is chosen from a stated rule, never tuned to make
    two surfaces agree (that would be the circularity trap).
  * KL needs the target strictly positive everywhere the other surface has
    mass, so `add_floor` puts a tiny epsilon floor on a surface before it
    is used as the second argument of a KL. Wasserstein needs no floor.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# --- Canonical representation constants --------------------------------
#
# These are the only knobs, and they are shared by every model. Change
# them in one place and both the baseline and the sampler move together.

# Physical size of one PDF cell. The extraction grid is ~6.67 m/cell over
# a 20 km square (~3000 cells); a location PDF for search planning does
# not need that: 100 m cells give a ~200 x 200 surface, which is the
# right granularity for probability-of-area and keeps KL / Wasserstein
# cheap. This is the "coarsened comparison" grid the Refined Approach doc
# refers to.
PDF_CELL_M = 100.0

# Gaussian smoothing length in metres. At 100 m cells this is 1.5 cells.
# Rationale to state: it is roughly a searcher's effective detection
# sweep width, so the smoothed surface reads as "probability the subject
# is within sensor range of this cell" rather than "probability in this
# exact 100 m box". Applied identically to ABM points and SNN spikes.
BANDWIDTH_M = 150.0

# Floor added before a surface is used as the denominator of a KL, as a
# fraction of a single uniform cell's mass. Keeps KL finite without
# meaningfully moving the surface.
KL_FLOOR_FRAC = 1e-6

# A PDF cell is treated as inaccessible (water: lake or river interior) and
# forced to zero probability if at least this fraction of the fine terrain
# cells it covers are inaccessible. 0.5 means "a majority-water cell is a
# water cell", which zeroes lake bodies without deleting the near-shore land
# in shoreline cells that are mostly dry.
WATER_FRAC_THRESH = 0.5


def coarsen_factor(cell_size_m: float, pdf_cell_m: float = PDF_CELL_M) -> int:
    """Integer number of terrain cells per PDF cell (at least 1)."""
    return max(1, int(round(float(pdf_cell_m) / float(cell_size_m))))


def grid_shape(terrain_shape, factor: int) -> tuple[int, int]:
    """PDF grid shape for a terrain of `terrain_shape` coarsened by `factor`."""
    h, w = terrain_shape
    return (int(np.ceil(h / factor)), int(np.ceil(w / factor)))


def cell_to_pdf_cell(cell, factor: int) -> tuple[int, int]:
    """Map a full-resolution (row, col) to its PDF (row, col)."""
    return (int(cell[0]) // factor, int(cell[1]) // factor)


def _normalise(surface: np.ndarray) -> np.ndarray:
    """Return a copy summing to 1. A flat all-zero surface becomes uniform."""
    s = surface.astype(np.float64)
    total = s.sum()
    if total <= 0:
        return np.full(s.shape, 1.0 / s.size, dtype=np.float64)
    return s / total


def points_to_pdf(
    points_rc: np.ndarray,
    terrain_shape,
    cell_size_m: float,
    pdf_cell_m: float = PDF_CELL_M,
    bandwidth_m: float = BANDWIDTH_M,
    inaccessible: np.ndarray | None = None,
    inaccessible_pdf: np.ndarray | None = None,
):
    """
    Build a normalised PDF from a cloud of continuous sample points.

    This is the ABM path: `points_rc` is the (n, 2) array of endpoint (or
    any) positions in full-resolution (row, col) coordinates, e.g.
    `result["final_cells"]`. The steps are exactly the "phi_ker" recipe:
    histogram the points onto the canonical coarse grid, blur with the
    shared Gaussian, normalise.

    Inaccessible cells (lake and river interiors) are forced to zero after
    smoothing and the surface renormalised, so no probability is left on
    water where the Gaussian blur bled it across a shoreline. Pass the
    region either pre-coarsened to the PDF grid (`inaccessible_pdf`,
    preferred - coarsen once and reuse across surfaces, and it lets the
    caller exempt known-accessible cells such as the find location) or as a
    full-resolution mask (`inaccessible`, coarsened here for convenience).

    Returns
    -------
    pdf : (ph, pw) float64, sums to 1
    info : dict with factor, pdf_cell_size_m, shape
    """
    factor = coarsen_factor(cell_size_m, pdf_cell_m)
    ph, pw = grid_shape(terrain_shape, factor)

    pts = np.asarray(points_rc)
    rows = np.clip((pts[:, 0] // factor).astype(np.int64), 0, ph - 1)
    cols = np.clip((pts[:, 1] // factor).astype(np.int64), 0, pw - 1)

    hist = np.bincount(rows * pw + cols, minlength=ph * pw).astype(np.float64)
    hist = hist.reshape(ph, pw)

    pdf = _finish(hist, factor, pdf_cell_m, bandwidth_m,
                  inaccessible, inaccessible_pdf, (ph, pw))
    info = {
        "factor": factor,
        "pdf_cell_size_m": float(cell_size_m) * factor,
        "shape": (ph, pw),
    }
    return pdf, info


def field_to_pdf(
    field: np.ndarray,
    src_cell_size_m: float,
    terrain_shape,
    terrain_cell_size_m: float,
    pdf_cell_m: float = PDF_CELL_M,
    bandwidth_m: float = BANDWIDTH_M,
    inaccessible: np.ndarray | None = None,
    inaccessible_pdf: np.ndarray | None = None,
):
    """
    Build a normalised PDF from an already-gridded, non-negative count
    field: the ABM occupancy grid (`density_frames[-1]`), or later the
    SNN's per-cell spike counts.

    This is the exact answer to the "can we do the same for the SNN?"
    question: yes. The only difference from `points_to_pdf` is that the
    counts already live on a grid, so there is no histogram step.

    The critical detail: the output grid is derived from the FULL-RESOLUTION
    terrain (`terrain_shape`, `terrain_cell_size_m`), NOT from the source
    field's own resolution. That guarantees this surface has exactly the
    same shape and physical footprint as `points_to_pdf` for the same
    incident, so an SNN field, an ABM occupancy field and an ABM endpoint
    field are all directly comparable cell-for-cell. The source field is
    first mapped up to terrain resolution, then coarsened to the canonical
    grid, then blurred with the SAME kernel and normalised.
    """
    factor = coarsen_factor(terrain_cell_size_m, pdf_cell_m)
    ph, pw = grid_shape(terrain_shape, factor)

    field = np.asarray(field, dtype=np.float64)
    ratio = max(1, int(round(float(src_cell_size_m) / float(terrain_cell_size_m))))
    if ratio > 1:
        field = np.repeat(np.repeat(field, ratio, axis=0), ratio, axis=1)
    # Conform to the exact terrain footprint (crop or pad), so the block
    # coarsening below lands on the canonical (ph, pw) grid.
    field = _fit_to_shape(field, terrain_shape)
    coarse = _block_sum(field, factor)
    coarse = _fit_to_shape(coarse, (ph, pw))

    pdf = _finish(coarse, factor, pdf_cell_m, bandwidth_m,
                  inaccessible, inaccessible_pdf, (ph, pw))
    info = {
        "factor": factor,
        "pdf_cell_size_m": float(terrain_cell_size_m) * factor,
        "shape": (ph, pw),
    }
    return pdf, info


def _fit_to_shape(arr: np.ndarray, shape) -> np.ndarray:
    """Crop or zero-pad the trailing edges so arr has exactly `shape`."""
    h, w = shape
    arr = arr[:h, :w]
    ph = h - arr.shape[0]
    pw = w - arr.shape[1]
    if ph or pw:
        arr = np.pad(arr, ((0, max(0, ph)), (0, max(0, pw))), mode="constant")
    return arr


def _finish(field, factor, pdf_cell_m, bandwidth_m,
            inaccessible=None, inaccessible_pdf=None, shape=None):
    """
    Blur, then zero any inaccessible PDF cells, then normalise.

    The inaccessible region may be supplied already coarsened to the PDF
    grid (`inaccessible_pdf`, used as-is) or at full resolution
    (`inaccessible`, coarsened here). The pre-coarsened form is preferred:
    it is coarsened once by the caller and shared across surfaces, and the
    caller can exempt known-accessible cells (e.g. the find location) first.
    """
    sigma_cells = float(bandwidth_m) / float(pdf_cell_m)
    if sigma_cells > 0:
        field = gaussian_filter(field, sigma=sigma_cells, mode="constant")

    mask = inaccessible_pdf
    if mask is None and inaccessible is not None:
        mask = coarsen_mask(inaccessible, factor, shape)
    if mask is not None:
        if mask.shape != field.shape:
            raise ValueError(
                f"inaccessible mask shape {mask.shape} != PDF grid {field.shape}"
            )
        field = field.copy()
        field[mask] = 0.0
    return _normalise(field)

def readout(field, inaccessible_pdf=None, pdf_cell_m=PDF_CELL_M,
            bandwidth_m=BANDWIDTH_M):
    """
    Belief from a non-negative count field ALREADY on the canonical PDF grid
    (e.g. the SNN's per-cell spike counts): blur with the shared Gaussian,
    zero inaccessible cells, normalise. The SNN's counterpart to field_to_pdf,
    using the identical kernel and mask so the two surfaces stay comparable.
    """
    field = np.asarray(field, dtype=np.float64)
    return _finish(field, 1, pdf_cell_m, bandwidth_m,
                   inaccessible_pdf=inaccessible_pdf, shape=field.shape)

def coarsen_mask(mask, factor, shape, thresh=WATER_FRAC_THRESH):
    """
    Coarsen a full-resolution boolean mask to the PDF grid. A PDF cell is
    True when at least `thresh` of the fine cells it covers are True.
    """
    frac = _block_sum(np.asarray(mask, dtype=np.float64), factor) / float(factor * factor)
    frac = _fit_to_shape(frac, shape)
    return frac >= thresh


def _block_sum(arr: np.ndarray, factor: int) -> np.ndarray:
    """Sum arr into non-overlapping factor x factor blocks (counts add)."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    ph = (-h) % factor
    pw = (-w) % factor
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)), mode="constant")
    H, W = arr.shape
    return arr.reshape(H // factor, factor, W // factor, factor).sum(axis=(1, 3))


def add_floor(pdf: np.ndarray, frac: float = KL_FLOOR_FRAC) -> np.ndarray:
    """
    Return a re-normalised copy with a small uniform floor, so it is safe
    as the denominator of a KL divergence. `frac` is measured against a
    single uniform cell (1 / n_cells).
    """
    floor = frac / pdf.size
    return _normalise(pdf + floor)