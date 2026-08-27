# tools/compare_snn_abm.py
import csv
import numpy as np

def load(path, key="search_cost_km2"):
    d = {}
    for r in csv.DictReader(open(path)):
        v = r.get(key)
        if v not in ("", "None", None):
            d[int(r["incident_index"])] = float(v)
    return d

abm = load("results/mc_baseline/mc_baseline.csv")
snn = load("results/snn/snn_metrics.csv")
common = sorted(set(abm) & set(snn))
a = np.array([abm[i] for i in common]); s = np.array([snn[i] for i in common])
wins = int((s <= a).sum())
print(f"incidents compared: {len(common)}")
print(f"median search cost   ABM {np.median(a):6.2f}   SNN {np.median(s):6.2f} km2")
print(f"mean   search cost   ABM {np.mean(a):6.2f}   SNN {np.mean(s):6.2f} km2")
print(f"SNN <= ABM on {wins}/{len(common)} incidents ({100*wins/len(common):.0f}%)")