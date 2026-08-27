"""
OSM feature extraction and rasterisation for SARsim.

Phase 0 changes from the original:
  * Geometries are reprojected into the DEM's metric CRS before
    rasterisation. pyrosm always returns EPSG:4326; rasterising that
    against a British National Grid transform would silently place every
    feature in the wrong location.
  * Land-cover POLYGON layers are added. The original extracted linear
    features and water only, which meant the KSTAT find-location column
    (Woodland / Building / Field / Beach ...) could not be calibrated
    against and detection probability could not be made canopy-dependent.
  * Ditches are split out of the streams layer, because KSTAT scores
    Ditch as its own find category.
  * A `barriers` layer is added for the KSTAT "Linear" category, which
    covers hedgerows, walls and fence lines rather than roads.

LAYER_TAXONOMY below is the explicit mapping from KSTAT find-location
categories to the layers produced here. Keep it accurate: it is the
correspondence the calibration stage will assert against, and it should
appear in the paper's data section.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyrosm import OSM
from rasterio.features import rasterize


# KSTAT find-location category -> layers in this module that represent it.
# 'Vehicle' has no spatial layer: it denotes being found in/at a vehicle,
# which is an attribute of the subject rather than a terrain class, and is
# handled at the behavioural-mode level instead.
LAYER_TAXONOMY = {
    "Woodland":     ["woodland"],
    "Building":     ["buildings"],
    "Road or Path": ["roads", "trails"],
    "Water":        ["river_interior", "lake_interior", "streams"],
    "Field":        ["field"],
    "Linear":       ["barriers", "powerlines", "railroads"],
    "Beach":        ["beach"],
    "Ditch":        ["ditches"],
    "Vehicle":      [],
}

LINEAR_FILTERS = {
    "roads": {"highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service",
    ]},
    "trails": {"highway": [
        "track", "footway", "path", "pedestrian", "steps",
        "bridleway", "via_ferrata",
    ]},
    "powerlines": {"power": ["line", "minor_line"]},
    "railroads": {"railway": [
        "rail", "light_rail", "tram", "subway", "narrow_gauge", "preserved",
    ]},
    "streams": {"waterway": ["stream", "drain", "river"]},
    "ditches": {"waterway": ["ditch"]},
    "barriers": {"barrier": [
        "hedge", "fence", "wall", "retaining_wall", "hedge_bank", "dry_stone_wall",
    ]},
}

AREA_FILTERS = {
    "woodland": {"natural": ["wood"], "landuse": ["forest"]},
    "field": {
        "landuse": ["farmland", "meadow", "grass", "orchard", "vineyard", "allotments"],
        "natural": ["grassland", "heath", "scrub"],
    },
    "beach": {"natural": ["beach", "shingle", "sand"]},
}

LINE_TYPES = ["LineString", "MultiLineString"]
POLY_TYPES = ["Polygon", "MultiPolygon"]


def extract_bbox(input_pbf: str | Path, bbox: list[float], output_pbf: str | Path) -> str:
    """
    Extract a bounding box from a master OSM PBF using osmium.

    `bbox` must be in WGS84 lon/lat -- use the `wgs84_envelope` returned by
    dem_processor.bng_square_bounds, which is guaranteed to contain the
    projected DEM extent even though a metric square is a rotated
    quadrilateral in lon/lat.

    The 'smart' strategy preserves complete geometries for features that
    cross the boundary.
    """
    if shutil.which("osmium") is None:
        print("[Feature Processor] Notice: 'osmium' CLI binary not found in system PATH.")
        print("[Feature Processor] Skipping CLI pre-clipping stage.")
        return str(input_pbf)

    bbox_str = ",".join(str(v) for v in bbox)
    cmd = [
        "osmium", "extract",
        "-b", bbox_str,
        str(input_pbf),
        "-s", "smart",
        "-o", str(output_pbf),
        "--overwrite",
    ]

    print(f"[Feature Processor] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Osmium extraction failed: {result.stderr}")

    print(f"[Feature Processor] Extracted region written to {output_pbf}")
    return str(output_pbf)


def _rasterize_gdf(gdf, geom_types, dem_shape, dem_transform, dem_crs):
    """
    Reproject to the DEM CRS and burn matching geometries into a uint8
    mask aligned to the DEM grid.

    The reprojection is the critical step: pyrosm hands back EPSG:4326,
    and rasterising lon/lat coordinates against a metric transform would
    put everything near the origin without raising anything.
    """
    shapes = []
    if gdf is not None and len(gdf) > 0:
        try:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if str(gdf.crs).upper() != str(dem_crs).upper():
                gdf = gdf.to_crs(dem_crs)
            valid = gdf[gdf.geometry.type.isin(geom_types)]
            shapes = [(geom, 1) for geom in valid.geometry if geom is not None and not geom.is_empty]
        except Exception as e:
            print(f"[Feature Processor] Reprojection/geometry error: {e}")
            shapes = []

    if not shapes:
        return np.zeros(dem_shape, dtype=np.uint8)

    return rasterize(
        shapes=shapes,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def _polygon_boundaries(gdf):
    """Exterior rings of polygons, as linear features."""
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[])
    out = gpd.GeoDataFrame(geometry=gdf.geometry.boundary)
    out.crs = gdf.crs
    return out


def extract_and_rasterize(
    pbf_path: str | Path,
    dem_path: str | Path,
    wgs84_envelope: list[float] | None = None,
) -> dict | None:
    """
    Extract OSM features by category and rasterise them onto the DEM grid.

    `wgs84_envelope` is the lon/lat box handed to pyrosm. It must be given
    explicitly now that the DEM is projected -- deriving it from the DEM
    bounds would hand pyrosm British National Grid eastings/northings.

    Returns a dict of {layer_name: uint8 mask}.
    """
    pbf_str = str(pbf_path)
    if not os.path.exists(pbf_str):
        print(f"[Feature Processor] PBF file not found: {pbf_str}")
        return None

    with rasterio.open(dem_path) as src:
        dem_shape = src.shape
        dem_transform = src.transform
        dem_crs = src.crs

    if wgs84_envelope is None:
        raise ValueError(
            "wgs84_envelope is required: the DEM is projected, so its bounds "
            "cannot be used as a lon/lat bounding box for pyrosm."
        )

    print("[Feature Processor] Initializing Pyrosm...")
    osm = OSM(pbf_str, bounding_box=list(wgs84_envelope), complete_relations=True)

    feature_matrices = {}

    def burn(gdf, geom_types):
        return _rasterize_gdf(gdf, geom_types, dem_shape, dem_transform, dem_crs)

    # --- Linear features -------------------------------------------------
    for layer_name, custom_filter in LINEAR_FILTERS.items():
        print(f"[Feature Processor] Extracting {layer_name}...")
        try:
            gdf = osm.get_data_by_custom_criteria(
                custom_filter=custom_filter,
                osm_keys_to_keep=list(custom_filter.keys()),
                keep_nodes=False,
                keep_ways=True,
                keep_relations=True,
            )
            feature_matrices[layer_name] = burn(gdf, LINE_TYPES)
        except Exception as e:
            print(f"[Feature Processor] {layer_name} extraction error: {e}")
            feature_matrices[layer_name] = np.zeros(dem_shape, dtype=np.uint8)

    # --- Land-cover polygons ---------------------------------------------
    # These carry the KSTAT find-location categories. Without them there is
    # nothing to calibrate the land-cover column against, and no basis for
    # making POD depend on canopy closure.
    for layer_name, custom_filter in AREA_FILTERS.items():
        print(f"[Feature Processor] Extracting {layer_name}...")
        try:
            gdf = osm.get_data_by_custom_criteria(
                custom_filter=custom_filter,
                osm_keys_to_keep=list(custom_filter.keys()),
                keep_nodes=False,
                keep_ways=True,
                keep_relations=True,
            )
            feature_matrices[layer_name] = burn(gdf, POLY_TYPES)
        except Exception as e:
            print(f"[Feature Processor] {layer_name} extraction error: {e}")
            feature_matrices[layer_name] = np.zeros(dem_shape, dtype=np.uint8)

    # --- Buildings --------------------------------------------------------
    print("[Feature Processor] Extracting buildings...")
    try:
        buildings_gdf = osm.get_buildings()
        feature_matrices["buildings"] = burn(buildings_gdf, POLY_TYPES)
    except Exception as e:
        print(f"[Feature Processor] buildings extraction error: {e}")
        feature_matrices["buildings"] = np.zeros(dem_shape, dtype=np.uint8)

    # --- Water polygons ---------------------------------------------------
    # Interiors are inaccessible regions; their boundaries are linear
    # features in their own right, exactly like a road or trail.
    print("[Feature Processor] Extracting water polygons...")
    try:
        water_gdf = osm.get_natural(custom_filter={"natural": ["water"]})
    except Exception as e:
        print(f"[Feature Processor] water polygon extraction error: {e}")
        water_gdf = None

    river_polys = gpd.GeoDataFrame()
    lake_polys = gpd.GeoDataFrame()
    if water_gdf is not None and len(water_gdf) > 0:
        water_gdf = water_gdf[water_gdf.geometry.type.isin(POLY_TYPES)]
        if "water" in water_gdf.columns:
            river_polys = water_gdf[water_gdf["water"].isin(["river", "riverbank"])]
            lake_polys = water_gdf[~water_gdf.index.isin(river_polys.index)]
        else:
            lake_polys = water_gdf

    feature_matrices["river_interior"] = burn(river_polys, POLY_TYPES)
    feature_matrices["lake_interior"] = burn(lake_polys, POLY_TYPES)
    feature_matrices["river_banks"] = burn(_polygon_boundaries(river_polys), LINE_TYPES)
    feature_matrices["lake_shorelines"] = burn(_polygon_boundaries(lake_polys), LINE_TYPES)

    print("[Feature Processor] Rasterization complete.")
    return feature_matrices


def summarise_coverage(feature_matrices: dict, cell_size_m: float) -> dict:
    """
    Phase 0 sanity check: fraction of the grid occupied by each layer, and
    for linear layers an implied total length.

    Read these before trusting anything downstream. An empty woodland layer
    in the Lake District, or a roads layer covering 40% of the grid, means
    the extraction or the reprojection is wrong -- and a silent CRS error
    looks exactly like an empty layer.
    """
    summary = {}
    for name, mask in feature_matrices.items():
        if mask is None:
            continue
        occupied = int(np.count_nonzero(mask))
        total = int(mask.size)
        summary[name] = {
            "cells": occupied,
            "fraction": occupied / total if total else 0.0,
            "approx_length_km": occupied * cell_size_m / 1000.0,
            "approx_area_km2": occupied * (cell_size_m ** 2) / 1e6,
        }
    return summary