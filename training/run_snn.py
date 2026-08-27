# training/run_snn.py
import argparse, csv, sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import snn_model as sm

COLUMNS = ["incident_index", "find_in_bounds", "lambda_m", "temperature", "tau",
           "n_chains", "burn_in", "readout", "pdf_cell_size_m", "entropy_bits",
           "n_modes", "peak_prob", "search_cost_km2", "prob_at_find",
           "dist_peak_to_find_m"]

def discover(data):
    out = []
    for p in sorted(Path(data).glob("incident_*")):
        if (p / "stack.npz").exists():
            try: idx = int(p.name.split("_")[1])
            except ValueError: idx = None
            out.append((idx, p))
    out.sort(key=lambda t: (t[0] is None, t[0])); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "processed"))
    ap.add_argument("--out",  default=str(PROJECT_ROOT / "results" / "snn"))
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--lambda-m", type=float, default=sm.LAMBDA_M)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out / "snn_heatmaps"; heatmap_dir.mkdir(parents=True, exist_ok=True)
    incs = discover(args.data)
    if args.only is not None:
        incs = [(i, p) for i, p in incs if i == args.only]
    rows = []
    for idx, path in incs:
        print(f"[SNN] incident {idx} ...", flush=True)
        r = sm.run_incident_snn(path, lambda_m=args.lambda_m)
        np.savez_compressed(out / f"incident_{idx}.npz", ...)          # (as before)
        sm.render_snn_heatmap(path, r["belief"], r["fields"]["factor"],
                              r["ipp_pdf"], r["find_pdf"],
                              heatmap_dir / f"incident_{idx}_snn.png", index=idx)
        rows.append(r["metrics"])

    with open(out / "snn_metrics.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nDone: {len(rows)} incidents. Metrics -> {out / 'snn_metrics.csv'}")

if __name__ == "__main__":
    main()