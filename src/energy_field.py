# src/energy_field.py
"""
Terrain energy field for Architecture 2 (Milestone 3).

Builds the static fields the spiking sampler needs, from the terrain stack
and the fixed PMF only (never from the baseline PDF):
  b : unary bias (log-density) = reachability (M3a) + traversal cost (M3b)
  W : pairwise coupling (terrain-shaped smoothness)          (M3c)

Built on the canonical 100 m PDF grid (pdf_common), so the sampled belief
lands on the same grid as the ABM baseline.
"""
import json
from pathlib import Path
import numpy as np
from skimage.graph import MCP_Geometric

import lp_model
import pdf_common as pc

import torch

J0               = 0.4    # base smoothness coupling on every edge
BARRIER_COUPLING = 0.0    # coupling across a water/cliff edge (belief may break)
PATH_BOOST       = 2.0    # coupling multiplier along a trail edge

# Reachability parameters, all stated (not fitted).
SLOPE_PENALTY    = 4.0     # extra cost per unit slope (m/m)
PATH_COST_FACTOR = 0.5     # movement this-times cheaper on a trail/road
BLOCKED_COST     = 1e6     # water: effectively impassable
REACH_LAMBDA_M   = 3000.0  # e-folding reach; from PMF dispersal x horizon
PATH_KEYS = ("trails", "roads", "railroads")

def terrain_fields(incident_dir):
    incident_dir = Path(incident_dir)
    terr = lp_model.Terrain.from_query_dir(incident_dir)
    ipp = tuple(json.load(open(incident_dir / "metadata.json"))["ipp_cell"])
    factor = pc.coarsen_factor(terr.cell_size_m)
    ph, pw = pc.grid_shape(terr.shape, factor)
    ones = np.ones(terr.shape, float)

    def block_mean(a):
        s = pc._block_sum(np.asarray(a, float), factor)
        n = pc._block_sum(ones, factor)
        return pc._fit_to_shape(s / np.maximum(n, 1), (ph, pw))

    def present(a):   # a fine 0/1 layer -> "present somewhere in this PDF cell"
        return pc._fit_to_shape(pc._block_sum(np.asarray(a, float), factor),
                                (ph, pw)) > 0

    with np.load(incident_dir / "stack.npz") as st:
        slope = block_mean(st["slope"])
        paths = np.zeros((ph, pw), bool)
        for k in PATH_KEYS:
            if k in st.files:
                paths |= present(st[k])

    water = pc.coarsen_mask(terr.inaccessible, factor, (ph, pw))
    return {"slope": slope, "water": water, "paths": paths,
            "ipp_pdf": (ipp[0] // factor, ipp[1] // factor),
            "cell_m": terr.cell_size_m * factor, "shape": (ph, pw),
            "factor": factor}


REACH_LOGSIGMA = 0.4538   # spread of ln(distance); the shape knob, set from SAR data


def reachability_bias(fields, lambda_m=REACH_LAMBDA_M, logsigma=REACH_LOGSIGMA):
    # lambda_m is now the MEDIAN find distance (exp of the log-normal's mu).
    cost = 1.0 + SLOPE_PENALTY * fields["slope"]
    cost[fields["paths"]] *= PATH_COST_FACTOR
    cost[fields["water"]] = BLOCKED_COST
    cost = np.ascontiguousarray(cost, float)
    dist, _ = MCP_Geometric(cost).find_costs([fields["ipp_pdf"]])
    dist = np.asarray(dist) * fields["cell_m"]
    reachable = np.isfinite(dist) & (dist < BLOCKED_COST * fields["cell_m"] * 0.5)

    mu = np.log(lambda_m)                             # median distance = exp(mu)
    d = np.clip(dist, fields["cell_m"] * 0.5, None)   # avoid ln(0) at the IPP
    b = np.full(fields["shape"], -50.0)
    ln_d = np.log(d[reachable])
    b[reachable] = -((ln_d - mu) ** 2) / (2.0 * logsigma ** 2) - ln_d
    return b, reachable

# --- M3b: traversal-cost bias ---
PATH_BONUS    = 1.0    # + log-density on trail/road cells
CLIFF_SLOPE   = 0.8    # m/m (~39 deg): steeper than this is a cliff
CLIFF_PENALTY = -6.0   # near-zero density on cliffs


def traversal_bias(fields):
    b = np.zeros(fields["shape"])
    b[fields["paths"]] += PATH_BONUS
    b[fields["slope"] > CLIFF_SLOPE] += CLIFF_PENALTY
    return b


def unary_bias(fields, lambda_m=REACH_LAMBDA_M):
    """The full unary bias: reachability envelope + traversal preferences."""
    b_reach, reachable = reachability_bias(fields, lambda_m)
    return b_reach + traversal_bias(fields), reachable


def coupling_fields(fields):
    """Directional per-edge couplings (Wh, Wv) as torch tensors."""
    sh = fields["shape"]
    blocked = fields["water"] | (fields["slope"] > CLIFF_SLOPE)
    paths = fields["paths"]
    Wh = np.full(sh, J0); Wv = np.full(sh, J0)
    m = np.zeros(sh, bool); m[:, :-1] = paths[:, :-1] & paths[:, 1:]; Wh[m] = J0 * PATH_BOOST
    m = np.zeros(sh, bool); m[:-1, :] = paths[:-1, :] & paths[1:, :]; Wv[m] = J0 * PATH_BOOST
    m = np.zeros(sh, bool); m[:, :-1] = blocked[:, :-1] | blocked[:, 1:]; Wh[m] = BARRIER_COUPLING
    m = np.zeros(sh, bool); m[:-1, :] = blocked[:-1, :] | blocked[1:, :]; Wv[m] = BARRIER_COUPLING
    return torch.from_numpy(Wh), torch.from_numpy(Wv)