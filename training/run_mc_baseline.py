"""
Phase 1 production run: MC baseline location PDFs + metrics for all 65
incidents. Run this once; everything downstream reads its outputs.

Outputs (under results/mc_baseline/):
  mc_baseline.csv                    one row per incident (all cases, one table)
  incident_<i>.npz                   pdf_endpoint, pdf_occupancy, ipp_pdf,
                                     find_pdf, pdf_cell_size_m - saved surfaces
  baseline_heatmaps/incident_<i>_heatmap.png   the heatmap image(s)

Usage:
  python training/run_mc_baseline.py                 # all incidents
  python training/run_mc_baseline.py --only 1        # just incident 1
  python training/run_mc_baseline.py --duration 24   # override elapsed time

Paths are resolved from this file's location, so it runs from anywhere.
Point --data at wherever the incident_<i> folders live if not the default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mc_baseline as mcb   # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "training"
DEFAULT_OUT = PROJECT_ROOT / "results" / "mc_baseline"

# Column order for the metrics CSV.
COLUMNS = [
    "incident_index", "find_in_bounds", "n_samples", "duration_hours", "seed",
    "cell_size_m", "pdf_cell_size_m", "total_steps",
    "entropy_bits", "n_modes", "peak_prob",
    "search_cost_km2", "prob_at_find", "dist_peak_to_find_m", "energy_statistic_m",
    "mean_displacement_m", "median_displacement_m", "p95_displacement_m",
    "effective_speed_ms", "area_covered_km2", "blocked_step_fraction",
    "endpoint_boundary_mass", "layers_missing_n",
    "op_count", "est_energy_j",
]


def discover_incidents(data_dir):
    dirs = []
    for p in sorted(Path(data_dir).glob("incident_*")):
        if (p / "stack.npz").exists():
            try:
                idx = int(p.name.split("_")[1])
            except ValueError:
                idx = None
            dirs.append((idx, p))
    dirs.sort(key=lambda t: (t[0] is None, t[0]))
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", type=int, default=None,
                    help="run a single incident index")
    ap.add_argument("--n-samples", type=int, default=mcb.N_SAMPLES)
    ap.add_argument("--duration", type=float, default=mcb.DURATION_HOURS)
    ap.add_argument("--heatmap-incident", type=int, default=None,
                    help="render ONLY this incident's heatmap (default: all)")
    ap.add_argument("--no-heatmaps", action="store_true",
                    help="skip heatmaps entirely (just surfaces + CSV)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Heatmap images live in their own subfolder; the metrics CSV and the
    # per-incident .npz surfaces stay in the parent (out) alongside it.
    heatmap_dir = out / "baseline_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    incidents = discover_incidents(args.data)
    if args.only is not None:
        incidents = [(i, p) for (i, p) in incidents if i == args.only]
    if not incidents:
        print(f"No incident_* folders with a stack.npz under {args.data}")
        sys.exit(1)

    rows = []
    for idx, path in incidents:
        print(f"[MC baseline] incident {idx} ...", flush=True)
        run = mcb.run_incident(
            path, n_samples=args.n_samples, duration_hours=args.duration,
        )
        # Persist the surfaces.
        np.savez_compressed(
            out / f"incident_{idx}.npz",
            pdf_endpoint=run["pdf_endpoint"].astype(np.float32),
            pdf_occupancy=run["pdf_occupancy"].astype(np.float32),
            pdf_cell_size_m=np.float32(run["pdf_cell_size_m"]),
            ipp_pdf=np.array(run["ipp_pdf"], np.int32),
            find_pdf=np.array(run["find_pdf"] if run["find_pdf"] else [-1, -1],
                              np.int32),
            primary=np.str_(run["pdf_primary_name"]),
        )
        rows.append(run["metrics"])

        # By default render one heatmap per incident into baseline_heatmaps/;
        # --heatmap-incident N restricts to a single one, --no-heatmaps skips.
        render_this = (not args.no_heatmaps) and (
            args.heatmap_incident is None or idx == args.heatmap_incident
        )
        if render_this:
            hp = mcb.render_heatmap(run, heatmap_dir / f"incident_{idx}_heatmap.png")
            print(f"           heatmap -> {hp}")

    # Write the single metric table for all cases, in the parent folder
    # (outside baseline_heatmaps).
    csv_path = out / "mc_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone: {len(rows)} incident(s).")
    print(f"Surfaces + metrics in {out}")
    print(f"Heatmaps in {heatmap_dir}")
    print(f"Metric table: {csv_path}")


if __name__ == "__main__":
    main()