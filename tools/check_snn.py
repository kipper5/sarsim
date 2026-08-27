import sys, torch
sys.path.insert(0, "src")
from snn_sampler import neuron_step, checkerboard_masks, sampler_step, run_sampler
from energy_field import terrain_fields, reachability_bias, unary_bias, CLIFF_SLOPE
import numpy as np
import math
from scipy.ndimage import uniform_filter
import numpy as np, torch
from energy_field import (terrain_fields, unary_bias, coupling_fields,
                          CLIFF_SLOPE, J0, PATH_BOOST, BARRIER_COUPLING)
from snn_sampler import run_sampler
import numpy as np, matplotlib

matplotlib.use("Agg");
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pdf_common as pc
from snn_model import run_incident_snn

import numpy as np, pdf_common as pc
from snn_model import run_incident_snn

def milestone0(biases=(-1.5, -0.5, 0.0, 0.8, 2.0), taus=(1,4), steps=60000, warm=2000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    b=torch.tensor(biases)
    for tau in taus:
        refrac = torch.zeros_like(b, dtype=torch.long)
        on = torch.zeros_like(b, dtype=torch.double)
        for t in range(steps):
            refrac = neuron_step(refrac, b, tau, gen)
            if t>=warm:
                on += (refrac>0).double()
        emp = on/(steps-warm)
        err = float((emp-torch.sigmoid(b)).abs().max())
        print(f"tau={tau}: max|emp - sigmoid(b)| = {err:.4f}  "
              f"({'PASS' if err < 0.01 else 'FAIL'})")

def exact_marginals(b, coupling):
    """Brute-force P(z_i = 1) by enumerating all 2^(H*W) states. Small grids only."""
    H, W = b.shape; N = H * W
    bf = b.flatten().double()
    edges = []
    for i in range(H):
        for j in range(W):
            k = i * W + j
            if j + 1 < W: edges.append((k, i * W + j + 1))
            if i + 1 < H: edges.append((k, (i + 1) * W + j))
    marg = torch.zeros(N, dtype=torch.double); Z = 0.0
    idx = torch.arange(N)
    for s in range(2 ** N):
        z = ((s >> idx) & 1).double()
        E = (bf * z).sum() + coupling * sum(z[a] * z[c] for a, c in edges)
        w = math.exp(E); Z += w; marg += w * z
    return (marg / Z).reshape(H, W)


def milestone1(size=4, coupling=0.6, taus=(1, 4), B=512, steps=4000, burn=1000, seed=0):
    torch.manual_seed(seed)
    b = torch.rand(size, size) * 2 - 1                 # biases in [-1, 1]
    red, black = checkerboard_masks(size, size)
    exact = exact_marginals(b, coupling)
    for tau in taus:
        gen = torch.Generator().manual_seed(1)
        refrac = torch.zeros(B, size, size, dtype=torch.long)
        acc = torch.zeros(size, size, dtype=torch.double); n = 0
        for t in range(steps):
            refrac = sampler_step(refrac, b, coupling, tau, red, black, gen)
            if t >= burn:
                acc += (refrac > 0).double().mean(0); n += 1
        err = float((acc / n - exact).abs().max())
        print(f"J={coupling} tau={tau}: max|emp - exact| = {err:.4f}  "
              f"({'PASS' if err < 0.02 else 'FAIL'})")


def target_to_bias(p, peak=0.8, eps=1e-6):
    """Uncoupled inversion: biases whose independent activations reproduce p*."""
    t = (p / p.max() * peak).clamp(eps, 1 - eps)
    return torch.log(t / (1 - t))


def milestone2(size=40, tau=1, seed=1):
    yy, xx = torch.meshgrid(torch.arange(size).double(),
                            torch.arange(size).double(), indexing="ij")
    bump = lambda cy, cx, s: torch.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * s * s))
    target = 1.0*bump(10, 12, 4) + 0.7*bump(28, 26, 5) + 0.5*bump(14, 30, 3)
    target = target / target.sum()

    gen = torch.Generator().manual_seed(seed)
    counts = run_sampler(target_to_bias(target), 0.0, tau, generator=gen)
    belief = counts / counts.sum()

    m = target > 0
    kl = float((target[m] * torch.log(target[m] / (belief[m] + 1e-12))).sum())
    print(f"M2 multimodal tau={tau}: KL(target||belief) = {kl:.4f}  "
          f"({'PASS' if kl < 0.01 else 'FAIL'})")

def milestone3a(incident_dir="data/processed/incident_1"):
    f = terrain_fields(incident_dir)
    b, reachable = reachability_bias(f)
    r0, c0 = f["ipp_pdf"]
    checks = {
        "dist zero at IPP (b max there)": abs(b[r0, c0] - b[reachable].max()) < 1e-6,
        "water is not reachable": not reachable[f["water"]].any(),
        "decays outward": b[r0, c0] > b[max(r0 - 20, 0), c0],
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    # Eyeball: import matplotlib.pyplot as plt; plt.imshow(b); plt.colorbar(); plt.show()

def milestone3b(incident_dir="data/processed/incident_1"):
    f = terrain_fields(incident_dir)
    b, reachable = unary_bias(f)
    lin = f["paths"]

    lift = float((b - uniform_filter(b, size=5))[lin & reachable].mean())
    t = np.exp(b - b.max()); t[~reachable] = 0; t[f["water"]] = 0; t /= t.sum()
    on_path = float(t[lin].sum()); frac = float(lin.mean())
    peak_on_path = bool(lin[np.unravel_index(t.argmax(), t.shape)])

    print(f"  path lift {lift:+.3f} (>0), mass on paths {on_path:.3f} vs "
          f"area {frac:.3f}, peak on path: {peak_on_path}")
    print(f"  [{'PASS' if lift > 0 and on_path > frac else 'FAIL'}] "
          f"paths are over-weighted; cliff active: {bool((f['slope']>CLIFF_SLOPE).any())}")

def milestone3c(incident_dir="data/processed/incident_1"):
    f = terrain_fields(incident_dir)
    Wh, Wv = coupling_fields(f)

    # (A) the coupling is exactly as specified (deterministic, noise-free)
    Whn = Wh.numpy(); sh = f["shape"]
    blk = f["water"] | (f["slope"] > CLIFF_SLOPE); pa = f["paths"]
    bar = np.zeros(sh, bool);  bar[:, :-1] = blk[:, :-1] | blk[:, 1:]
    pth = np.zeros(sh, bool);  pth[:, :-1] = pa[:, :-1] & pa[:, 1:]; pth &= ~bar
    base = np.zeros(sh, bool);  base[:, :-1] = True; base &= ~bar & ~pth
    okA = (np.allclose(Whn[bar], BARRIER_COUPLING)
           and np.allclose(Whn[pth], J0 * PATH_BOOST)
           and np.allclose(Whn[base], J0))
    print(f"  (A) structure [{'PASS' if okA else 'FAIL'}]: barrier {Whn[bar].mean():.2f}, "
          f"path {Whn[pth].mean():.2f}, base {Whn[base].mean():.2f}")

    # (B) two independent samples of the field must agree -> converged
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = torch.from_numpy(unary_bias(f)[0]).to(dev)
    Wh, Wv, water = Wh.to(dev), Wv.to(dev), f["water"]
    def belief(seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        c = run_sampler(b, (Wh, Wv), 1, n_chains=64, burn_in=1500,
                        readout=3000, generator=g).cpu().numpy()
        c[water] = 0; return c / c.sum()
    a, b2 = belief(1), belief(2)
    tv = float(0.5 * np.abs(a - b2).sum())
    print(f"  (B) convergence [{'PASS' if tv < 0.05 else 'FAIL'}]: TV(run1, run2) = {tv:.3f}")

def milestone3d(incident_index=1, temperature=1.0,
                data_root="data/processed", results="results/mc_baseline"):

    r = run_incident_snn(f"{data_root}/incident_{incident_index}",
                         temperature=temperature)
    belief, f = r["belief"], r["fields"]
    base = np.load(f"{results}/incident_{incident_index}.npz")["pdf_occupancy"].astype(float)
    base /= base.sum()

    q = pc.add_floor(belief)
    kl = float((base[base > 0] * np.log(base[base > 0] / q[base > 0])).sum())
    ent = lambda x: float(-(x[x > 0] * np.log2(x[x > 0])).sum())
    print(f"  beta={temperature}: KL(base||SNN)={kl:.3f}  "
          f"entropy SNN={ent(belief):.2f}  baseline={ent(base):.2f} bits")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (s, name) in zip(ax, [(base, "ABM baseline"), (belief, "SNN belief")]):
        m = np.ma.masked_where(s < s.max() * 1e-4, s)
        a.imshow(m, cmap="magma", norm=LogNorm(vmin=s.max() / 1e4, vmax=s.max()))
        a.scatter([f["ipp_pdf"][1]], [f["ipp_pdf"][0]], marker="*", s=160,
                  c="black", edgecolor="white"); a.set_title(name)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Incident {incident_index}  -  KL(base||SNN)={kl:.2f}")
    fig.savefig(f"{results}/snn_incident_{incident_index}.png",
                bbox_inches="tight", dpi=130)

def tune_lambda(indices=(1, 5, 20, 40, 63), lambdas=(1000, 1500, 2000, 3000, 4500),
                data_root="data/processed", results="results/mc_baseline",
                burn_in=1000, readout=1500):
    ent = lambda x: float(-(x[x > 0] * np.log2(x[x > 0])).sum())
    rows = []
    print(f"{'lambda_m':>9} {'med gap(bits)':>13} {'med KL':>8}")
    for lam in lambdas:
        gaps, kls = [], []
        for i in indices:
            bel = run_incident_snn(f"{data_root}/incident_{i}", lambda_m=lam,
                                   burn_in=burn_in, readout=readout)["belief"]
            base = np.load(f"{results}/incident_{i}.npz")["pdf_occupancy"].astype(float)
            base /= base.sum()
            gaps.append(ent(bel) - ent(base))
            q = pc.add_floor(bel)
            kls.append(float((base[base > 0] * np.log(base[base > 0] / q[base > 0])).sum()))
        mg, mk = float(np.median(gaps)), float(np.median(kls))
        rows.append((lam, mg, mk))
        print(f"{lam:>9} {mg:>13.2f} {mk:>8.3f}")
    best = min(rows, key=lambda r: abs(r[1]))
    print(f"\nOperating point: lambda = {best[0]} m  "
          f"(median spread gap {best[1]:+.2f} bits, median KL {best[2]:.3f})")
    return best[0]

def tune_lambda_intense(indices=(1, 5, 12, 20, 28, 40, 52, 60, 63),
                        lambdas=tuple(range(600, 2600, 100)),
                        betas=(1.0,),
                        data_root="data/processed", results="results/mc_baseline",
                        burn_in=1500, readout=3000, seed=1):
    ent = lambda x: float(-(x[x > 0] * np.log2(x[x > 0])).sum())

    # cache each incident's baseline surface + entropy once
    base = {}
    for i in indices:
        b = np.load(f"{results}/incident_{i}.npz")["pdf_occupancy"].astype(float)
        base[i] = (b / b.sum(), ent(b / b.sum()))

    rows = []
    print(f"{'beta':>5} {'lambda':>7} {'med gap':>9} {'med KL':>8}")
    for beta in betas:
        for lam in lambdas:
            gaps, kls = [], []
            for i in indices:
                bel = run_incident_snn(f"{data_root}/incident_{i}", lambda_m=lam,
                                       temperature=beta, burn_in=burn_in,
                                       readout=readout, seed=seed)["belief"]
                bpdf, bent = base[i]
                gaps.append(ent(bel) - bent)
                q = pc.add_floor(bel)
                kls.append(float((bpdf[bpdf > 0] * np.log(bpdf[bpdf > 0] / q[bpdf > 0])).sum()))
            row = (beta, lam, float(np.median(gaps)), float(np.median(kls)))
            rows.append(row)
            print(f"{row[0]:>5} {row[1]:>7} {row[2]:>+9.2f} {row[3]:>8.3f}")

    op = min(rows, key=lambda r: abs(r[2]))              # entropy-matched: the operating point
    kmin = min(rows, key=lambda r: r[3])                 # KL floor: a diagnostic only
    print(f"\nOPERATING POINT (entropy-matched): beta={op[0]} lambda={op[1]}  "
          f"gap {op[2]:+.2f} bits, KL {op[3]:.3f}")
    print(f"(diagnostic) KL floor at beta={kmin[0]} lambda={kmin[1]} KL={kmin[3]:.3f} "
          f"gap {kmin[2]:+.2f}  <- do NOT adopt this as the operating point")
    return op[1]

if __name__ == "__main__":
    tune_lambda_intense()