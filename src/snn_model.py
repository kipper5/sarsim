# src/snn_model.py  (replace the file with this fuller version)
"""
SNN belief per incident (Architecture 2, Milestones 3d + 5).

Samples the terrain energy field at the fixed operating point, reads the
belief out through pdf_common onto the ABM baseline's grid, and scores it
against ground truth with the ABM's own metric code, so the two models are
directly comparable. Parallels mc_baseline.run_incident.
"""
import json
from pathlib import Path
import numpy as np
import torch

import pdf_common as pc
import energy_field as ef
from snn_sampler import run_sampler
from mc_baseline import (_search_cost_km2, _dist_peak_to_find_m,
                         _entropy_bits, _count_modes)   # SAME metrics as the ABM
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.lines as mlines, matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import sim_render
from mc_baseline import DEFAULT_LAYER_STYLE


# Operating point from the lambda sweep: one global value, stated, not fitted.
LAMBDA_M    = 1000.0
TEMPERATURE = 1.0
TAU         = 1
N_CHAINS    = 64
BURN_IN     = 1500
READOUT     = 3000


def _truth_cells(incident_dir, factor, shape):
    meta = json.load(open(Path(incident_dir) / "metadata.json"))
    ipp = meta["ipp_cell"]
    ipp_pdf = (int(ipp[0]) // factor, int(ipp[1]) // factor)
    fc, fib = meta.get("find_cell"), meta.get("find_in_bounds")
    find_pdf = None
    if fc is not None and fib is not False:
        fp = (int(fc[0]) // factor, int(fc[1]) // factor)
        ph, pw = shape
        if 0 <= fp[0] < ph and 0 <= fp[1] < pw:
            find_pdf = fp
    return meta.get("incident_index"), ipp_pdf, find_pdf


def run_incident_snn(incident_dir, lambda_m=LAMBDA_M, temperature=TEMPERATURE,
                     tau=TAU, n_chains=N_CHAINS, burn_in=BURN_IN, readout=READOUT,
                     seed=1, device=None):
    f = ef.terrain_fields(incident_dir)
    factor = f["factor"]
    idx, ipp_pdf, find_pdf = _truth_cells(incident_dir, factor, f["shape"])

    # exempt the known-accessible IPP and find from the water mask (as the ABM does)
    water = f["water"].copy(); water[ipp_pdf] = False
    if find_pdf is not None:
        water[find_pdf] = False

    b_np, _ = ef.unary_bias(f, lambda_m=lambda_m)
    Wh, Wv = ef.coupling_fields(f)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    beta = float(temperature)
    b = torch.from_numpy(b_np * beta).to(dev)
    Wh, Wv = (Wh * beta).to(dev), (Wv * beta).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    counts = run_sampler(b, (Wh, Wv), tau, n_chains=n_chains, burn_in=burn_in,
                         readout=readout, generator=g).cpu().numpy()

    belief = pc.readout(counts, inaccessible_pdf=water)
    cell_km2 = (f["cell_m"] / 1000.0) ** 2

    m = {"incident_index": idx, "find_in_bounds": find_pdf is not None,
         "lambda_m": lambda_m, "temperature": beta, "tau": tau,
         "n_chains": n_chains, "burn_in": burn_in, "readout": readout,
         "pdf_cell_size_m": f["cell_m"], "entropy_bits": _entropy_bits(belief),
         "n_modes": _count_modes(belief), "peak_prob": float(belief.max())}
    if find_pdf is not None:
        m["search_cost_km2"] = _search_cost_km2(belief, find_pdf, cell_km2)
        m["prob_at_find"] = float(belief[find_pdf])
        m["dist_peak_to_find_m"] = _dist_peak_to_find_m(belief, find_pdf, f["cell_m"])
    else:
        for k in ("search_cost_km2", "prob_at_find", "dist_peak_to_find_m"):
            m[k] = None

    return {"belief": belief, "counts": counts, "fields": f,
            "ipp_pdf": ipp_pdf, "find_pdf": find_pdf, "metrics": m}

def render_snn_heatmap(incident_dir, belief, factor, ipp_pdf, find_pdf, out_path,
                       index=None, cmap="magma", log_decades=4.0, dpi=150):
    with np.load(Path(incident_dir) / "stack.npz") as st:
        layers = {k: st[k] for k in st.files if st[k].ndim == 2}
    H, W = next(v.shape for v in layers.values())
    grad = layers.get("gradient_edges", np.zeros((H, W), np.uint8))
    bg, legend = sim_render.composite_layers(DEFAULT_LAYER_STYLE, layers, grad)

    up = np.repeat(np.repeat(belief, factor, 0), factor, 1)[:H, :W]
    vmax = float(up.max()); vmin = vmax / (10 ** log_decades)
    masked = np.ma.masked_where(up < vmin, up)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    ax.imshow(bg, interpolation="nearest")
    im = ax.imshow(masked, cmap=cmap, alpha=0.72, norm=LogNorm(vmin=vmin, vmax=vmax),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "probability mass per PDF cell (log scale)")

    ir, ic = ipp_pdf[0]*factor + factor//2, ipp_pdf[1]*factor + factor//2
    ax.scatter([ic], [ir], marker="*", s=280, facecolor="black",
               edgecolor="white", linewidths=1.3, zorder=10)
    handles = [mlines.Line2D([], [], marker="*", linestyle="None", markersize=15,
               markerfacecolor="black", markeredgecolor="white", label="IPP")]
    if find_pdf is not None:
        fr, fc = find_pdf[0]*factor + factor//2, find_pdf[1]*factor + factor//2
        ax.scatter([fc], [fr], marker="X", s=200, facecolor="lime",
                   edgecolor="black", linewidths=1.3, zorder=11)
        handles.append(mlines.Line2D([], [], marker="X", linestyle="None", markersize=13,
                       markerfacecolor="lime", markeredgecolor="black", label="Found location"))
    handles += [mpatches.Patch(color=np.array(c)/255., label=l) for l, c in legend]

    ax.set_title(f"Incident {index}  -  SNN belief")
    ax.set_xlim(0, W-1); ax.set_ylim(H-1, 0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    loc = MaxNLocator(nbins=6, integer=True, steps=[1, 2, 2.5, 5, 10])
    lat = [t for t in loc.tick_values(0, H-1) if 0 <= t <= H-1]
    ax.set_yticks([(H-1)-t for t in lat]); ax.set_yticklabels([str(int(round(t))) for t in lat])
    ax.set_xlabel("Longitude (x cells)"); ax.set_ylabel("Latitude (y cells)")
    for n, sp in ax.spines.items(): sp.set_visible(n in ("left", "bottom"))
    ncol = min(4, max(2, (len(handles)+3)//4))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=ncol, fontsize=9, framealpha=0.9)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return str(out_path)