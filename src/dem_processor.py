"""
DEM acquisition and topographic feature extraction for SARsim.

Phase 0 changes from the original:
  * Output is reprojected to a METRIC CRS (EPSG:27700, British National
    Grid) instead of staying in EPSG:4326 degrees. Every downstream
    physics stage needs x and y in metres so that the diffusion tensor is
    isotropic and gradients carry real units (m/m rather than m/pixel).
  * Grid size is derived from an explicit `resolution_m` rather than a
    hardcoded 3000x3000, and the achieved cell size is returned so it can
    be recorded in metadata and read by every later stage.
  * `extract_dem_features` returns the continuous elevation and slope
    fields alongside the Canny edge mask. The edge mask is what the
    static map draws; the continuous fields are what the drift potential
    will be built from in Phase 1.

The Canny parameters and compositing behaviour are unchanged, so the
rendered elevation layer is visually identical to the reference pipeline.
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from skimage.feature import canny

# British National Grid. Metric, and the natural choice for UK-wide work.
# For a non-UK deployment swap this for the local UTM zone; nothing else
# in the pipeline assumes BNG specifically.
TARGET_CRS = "EPSG:27700"
SOURCE_CRS = "EPSG:4326"

# Guard rails on the output grid. The lower bound stops tiny queries from
# producing a grid too coarse to resolve linear features; the upper bound
# stops a 25 km query at 1 m/cell from trying to allocate 50000^2 cells.
MIN_GRID_PX = 256
MAX_GRID_PX = 8000


def bng_square_bounds(lat, lon, radius_m, target_crs=TARGET_CRS, pad_deg=0.01):
    """
    Convert a centre point and radius into an exactly square extent in
    projected metres, plus a WGS84 envelope guaranteed to contain it.

    The original pipeline built a lat/lon bbox and relied on a cos(lat)
    correction to keep it square in metres. That works, but it leaves the
    raster indexed by degrees. Projecting the centre first and then
    stepping +/- radius in metres gives a genuinely square metric extent
    with no trigonometry and no latitude-dependent distortion.

    A square in BNG maps to a slightly rotated quadrilateral in lon/lat,
    so the WGS84 envelope is the bounding box of the four transformed
    corners plus a small pad. That envelope is what osmium and pyrosm
    should clip against, so that OSM coverage strictly contains the DEM.

    Returns
    -------
    proj_bounds : (min_x, min_y, max_x, max_y) in target_crs metres
    wgs84_envelope : [min_lon, min_lat, max_lon, max_lat]
    """
    fwd = Transformer.from_crs(SOURCE_CRS, target_crs, always_xy=True)
    inv = Transformer.from_crs(target_crs, SOURCE_CRS, always_xy=True)

    cx, cy = fwd.transform(lon, lat)
    proj_bounds = (cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)

    corner_x = [proj_bounds[0], proj_bounds[2], proj_bounds[0], proj_bounds[2]]
    corner_y = [proj_bounds[1], proj_bounds[1], proj_bounds[3], proj_bounds[3]]
    lons, lats = inv.transform(corner_x, corner_y)

    wgs84_envelope = [
        min(lons) - pad_deg,
        min(lats) - pad_deg,
        max(lons) + pad_deg,
        max(lats) + pad_deg,
    ]
    return proj_bounds, wgs84_envelope


def resolve_grid(proj_bounds, resolution_m):
    """
    Work out the output grid size for a requested cell size, clamped to
    the guard rails. Returns (shape, achieved_resolution_m).

    The achieved resolution can differ slightly from the request because
    the grid must be a whole number of cells; it can differ a lot if the
    request would breach a guard rail. Either way the achieved value is
    what gets written to metadata, and that is the number every
    downstream stage must use.
    """
    span_x = proj_bounds[2] - proj_bounds[0]
    span_y = proj_bounds[3] - proj_bounds[1]

    width = int(round(span_x / float(resolution_m)))
    height = int(round(span_y / float(resolution_m)))

    width = max(MIN_GRID_PX, min(MAX_GRID_PX, width))
    height = max(MIN_GRID_PX, min(MAX_GRID_PX, height))

    achieved = span_x / width
    return (height, width), achieved


def extract_dem_bbox(
    cop30_dir: str | Path,
    wgs84_envelope: list[float],
    proj_bounds: tuple[float, float, float, float],
    output_path: str | Path,
    resolution_m: float = 10.0,
    target_crs: str = TARGET_CRS,
) -> dict | None:
    """
    Find intersecting COP30 tiles, merge them, and reproject onto a
    square metric grid covering `proj_bounds`.

    COP30 tiles are ~30 m/px in EPSG:4326. We merge in the native CRS
    (cheap, no resampling) and then do a single reproject+resample into
    the target grid, so elevation is interpolated exactly once.

    Returns a dict of grid information on success, or None on failure.
    """
    min_lon, min_lat, max_lon, max_lat = wgs84_envelope
    cop30_path = Path(cop30_dir)

    if not cop30_path.exists():
        print(f"[DEM Processor] Directory does not exist: {cop30_dir}")
        return None

    tif_files = list(cop30_path.rglob("*.tif"))
    if not tif_files:
        print(f"[DEM Processor] No .tif files found in {cop30_dir}")
        return None

    open_datasets = []
    for tif in tif_files:
        try:
            src = rasterio.open(tif)
            b = src.bounds
            if not (b.left > max_lon or b.right < min_lon or b.top < min_lat or b.bottom > max_lat):
                open_datasets.append(src)
            else:
                src.close()
        except Exception as e:
            print(f"[DEM Processor] Could not read raster {tif}: {e}")

    if not open_datasets:
        print("[DEM Processor] No overlapping DEM tiles found for the bounding box.")
        return None

    try:
        mosaic, mosaic_transform = merge(
            open_datasets, bounds=(min_lon, min_lat, max_lon, max_lat)
        )
        src_crs = open_datasets[0].crs
        src_nodata = open_datasets[0].nodata

        target_shape, achieved_res = resolve_grid(proj_bounds, resolution_m)
        target_height, target_width = target_shape
        target_transform = from_bounds(
            proj_bounds[0], proj_bounds[1], proj_bounds[2], proj_bounds[3],
            target_width, target_height,
        )

        resampled = np.zeros((mosaic.shape[0], target_height, target_width), dtype=np.float32)
        reproject(
            source=mosaic,
            destination=resampled,
            src_transform=mosaic_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        # COP30 marks sea as nodata. Left as NaN it would poison the
        # Gaussian smoothing and gradient in Phase 1, so fill with 0 m
        # (sea level), which is both physically sensible and a hard
        # inaccessible region the drift model will exclude anyway.
        nan_fraction = float(np.isnan(resampled).mean())
        if nan_fraction > 0:
            print(f"[DEM Processor] Filling {nan_fraction:.2%} nodata cells with 0 m (sea level).")
            resampled = np.nan_to_num(resampled, nan=0.0)

        out_meta = {
            "driver": "GTiff",
            "count": resampled.shape[0],
            "dtype": "float32",
            "height": target_height,
            "width": target_width,
            "transform": target_transform,
            "crs": target_crs,
            "compress": "deflate",
        }
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(resampled)

        print(
            f"[DEM Processor] Saved DEM ({target_width}x{target_height} @ "
            f"{achieved_res:.2f} m/cell, {target_crs}) to {output_path}"
        )

        return {
            "shape": [target_height, target_width],
            "cell_size_m": achieved_res,
            "crs": target_crs,
            "proj_bounds": list(proj_bounds),
            "nodata_fraction": nan_fraction,
        }

    except Exception as e:
        print(f"[DEM Processor] Error processing DEM raster: {e}")
        return None
    finally:
        for src in open_datasets:
            src.close()


def extract_dem_features(
    dem_path: str | Path,
    cell_size_m: float | None = None,
    smooth_sigma_px: float = 2.0,
    smooth_sigma_m: float | None = None,
) -> dict:
    """
    Extract topographic structure from the DEM.

    Rendering path (unchanged from the reference pipeline):
      1. Gaussian-smooth elevation (sigma=2 px, matching imgaussfilt)
      2. Gradient magnitude by central differences (matching imgradient)
      3. Percentile-clip normalise, then Canny with thresholds [0.01, 0.3]

    Canny on the *gradient* rather than on raw elevation is deliberate:
    edges in the slope field pick out ridgelines and drainage channels,
    whereas edges in the elevation field are just contours.

    Phase 0 additions:
      * `cell_size_m` scales the gradient into true slope (m/m). This has
        no effect on the Canny output, because the normalisation step
        divides the scaling straight back out -- the edge mask, and so
        the static map, is bit-identical either way. It matters because
        the returned `slope` field is what Phase 1 builds drift from, and
        that must be in real units.
      * Elevation and slope are returned so they can be persisted once
        rather than recomputed by every consumer.

    Note on `smooth_sigma_px`: sigma is in PIXELS, so the physical
    smoothing length depends on cell size. The reference used sigma=2 at
    ~6.67 m/cell, i.e. ~13 m. Pass `smooth_sigma_m` to hold the physical
    length fixed across resolutions instead; leave it None to reproduce
    the reference look exactly.

    Returns
    -------
    dict with keys 'edges' (uint8 mask), 'elevation' (float32, m),
    'slope' (float32, m/m or m/px if cell_size_m is None)
    """
    with rasterio.open(dem_path) as src:
        dem_array = src.read(1).astype(np.float32)
        if cell_size_m is None:
            cell_size_m = abs(src.transform.a)

    sigma = smooth_sigma_px
    if smooth_sigma_m is not None:
        sigma = max(0.5, smooth_sigma_m / float(cell_size_m))

    # 1. Gaussian smoothing
    dem_smooth = gaussian_filter(dem_array, sigma=sigma)

    # 2. Gradient magnitude. Spacing arguments convert rise-per-pixel into
    # rise-per-metre, giving dimensionless slope.
    dy, dx = np.gradient(dem_smooth, cell_size_m, cell_size_m)
    gradient_mag = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)

    # Normalise to [0, 1] for Canny thresholding. Slope fields are
    # long-tailed: a few cliff pixels can be an order of magnitude above
    # the rest, and plain min-max lets them set the ceiling, crushing the
    # useful mid-range signal towards zero. Clipping at the 2nd-98th
    # percentiles keeps the normalisation robust to those outliers.
    p_low, p_high = np.percentile(gradient_mag, [2, 98])
    if p_high > p_low:
        grad_norm = np.clip((gradient_mag - p_low) / (p_high - p_low), 0, 1)
    else:
        grad_norm = np.zeros_like(gradient_mag)

    # 3. Canny on the normalised gradient field. Smoothing already
    # happened in step 1, so the internal sigma is minimal (skimage
    # requires sigma > 0).
    edges = canny(grad_norm, sigma=0.1, low_threshold=0.01, high_threshold=0.3)

    return {
        "edges": edges.astype(np.uint8),
        "elevation": dem_array,
        "slope": gradient_mag,
    }