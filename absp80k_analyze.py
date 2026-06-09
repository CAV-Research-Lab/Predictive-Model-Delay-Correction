#!/usr/bin/env python3
"""Comprehensive ABSP vs SBSP vs A-SAC vs SAC comparison on FetchSlide @250-290ms, 80k, 8 seeds.

Authoritative metric: track/step_reward (true euclidean tracking error) -> cm.
Robust to reboot/rerun: per seed picks the SAC_* run with the most steps.
Outputs (1) final-performance table + convergence, (2) IQM[IQR] at training checkpoints,
(3) ordering vs hypothesis, and (with --plot) an IQM+IQR training-curve figure.

  python3 absp80k_analyze.py            # tables only (fast)
  python3 absp80k_analyze.py --plot     # + absp80k_training_curves.png
"""
import sys, numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

DELAY, SEEDS, CONV_CM, GOAL_CM = "250-290ms", range(8), 15.0, 5.0
CKPTS = [20000, 40000, 60000, 80000]
METHODS = [("ABSP",  "runs_FetchSlide_80k_absp", "PMDC"),
           ("SBSP",  "runs_FetchSlide_80k",      "PMDC"),
           ("A-SAC", "runs_FetchSlide_80k",      "A-SAC"),
           ("SAC",   "runs_FetchSlide_80k",      "SAC")]

def load_series(base, alg, seed):
    sd = Path(base) / "pred_logs" / alg / "5" / DELAY / f"seed_{seed}"
    if not sd.exists():
        return None
    best = None
    for sub in sorted(sd.glob("SAC_*")):
        ea = event_accumulator.EventAccumulator(str(sub)); ea.Reload()
        if "track/step_reward" not in ea.Tags().get("scalars", []):
            continue
        sc = ea.Scalars("track/step_reward")
        if not sc:
            continue
        steps = np.array([e.step for e in sc]); vals = np.abs(np.array([e.value for e in sc])) * 100.0
        if best is None or steps[-1] > best[0][-1]:
            best = (steps, vals)
    return best

def windowed_err(series, step, window=4000):
    """Mean err over (step-window, step]; None if the seed has not reached ~step."""
    steps, vals = series
    if steps[-1] < step - window:        # seed nowhere near this checkpoint yet
        return None
    m = (steps <= step) & (steps > step - window)
    if not m.any():
        return None
    return float(vals[m].mean())

def iqm_iqr(xs):
    x = np.sort(np.asarray([v for v in xs if v is not None], float))
    n = len(x)
    if n == 0:
        return None
    iqm = x[n // 4: n - n // 4].mean() if n >= 4 else x.mean()
    p25, p75 = np.percentile(x, [25, 75])
    return iqm, p25, p75, n

# ---- load everything ----
DATA = {(lab, s): load_series(b, a, s) for lab, b, a in METHODS for s in SEEDS}

def final_err(series):           # last logged 100-window
    steps, vals = series
    return float(vals[-100:].mean()), int(steps[-1])

print("=" * 78)
print("ABSP vs SBSP vs A-SAC vs SAC | FetchSlide-RemotePDNorm-v0 @250-290ms | 80k | 8 seeds")
print("=" * 78)

# ---- Table 1: final performance + convergence (completed seeds, >=79k) ----
print("\n[1] FINAL PERFORMANCE  (completed seeds only; tracking error in cm, lower=better)")
print(f"    {'method':6s} {'done':>4s} {'conv':>6s}  {'IQM':>5s} {'IQR(25-75)':>12s} {'median':>6s} {'mean+-std':>11s}")
final_iqm = {}
for lab, b, a in METHODS:
    comp = []
    for s in SEEDS:
        ser = DATA[(lab, s)]
        if ser is None:
            continue
        fe, st = final_err(ser)
        if st >= 79000:
            comp.append(fe)
    if comp:
        r = iqm_iqr(comp); v = np.array(comp)
        conv = int((v < CONV_CM).sum())
        final_iqm[lab] = r[0]
        print(f"    {lab:6s} {len(comp):>4d} {conv:>3d}/{len(comp):<2d}  {r[0]:>5.1f} {f'[{r[1]:.1f}-{r[2]:.1f}]':>12s} "
              f"{np.median(v):>6.1f} {f'{v.mean():.1f}+-{v.std():.1f}':>11s}")
    else:
        print(f"    {lab:6s} {0:>4d}     -       -            -      -       -")

# ---- Table 2: per-seed final error grid ----
print("\n[2] PER-SEED final error (cm)   [*]=converged<15   [~]=still running")
hdr = "    seed:  " + "".join(f"{s:>7d}" for s in SEEDS)
print(hdr)
for lab, b, a in METHODS:
    cells = []
    for s in SEEDS:
        ser = DATA[(lab, s)]
        if ser is None:
            cells.append(f"{'-':>7s}"); continue
        fe, st = final_err(ser)
        mark = "~" if st < 79000 else ("*" if fe < CONV_CM else " ")
        cells.append(f"{fe:>6.1f}{mark}")
    print(f"    {lab:6s} " + "".join(cells))

# ---- Table 3: IQM[IQR] across training checkpoints ----
print("\n[3] IQM [IQR 25-75] vs TRAINING STEP  (cm; n = seeds reaching that step)")
print(f"    {'method':6s} " + "".join(f"{str(c//1000)+'k':>16s}" for c in CKPTS))
for lab, b, a in METHODS:
    row = []
    for c in CKPTS:
        vals = [windowed_err(DATA[(lab, s)], c) for s in SEEDS if DATA[(lab, s)] is not None]
        r = iqm_iqr(vals)
        row.append("-".rjust(16) if r is None else f"{r[0]:.1f}[{r[1]:.0f}-{r[2]:.0f}]n{r[3]}".rjust(16))
    print(f"    {lab:6s} " + "".join(row))

# ---- ordering check ----
if final_iqm:
    order = sorted(final_iqm, key=final_iqm.get)
    print(f"\n[4] Ranking by final IQM (best->worst): {' < '.join(order)}")
    print(f"    Hypothesis: ABSP < SBSP < A-SAC < SAC   ({'MATCH' if order == ['ABSP','SBSP','A-SAC','SAC'] else 'does NOT match yet'})")

# ---- optional plot ----
if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    grid = np.arange(2000, 80001, 2000)
    colors = {"ABSP": "#1b7837", "SBSP": "#762a83", "A-SAC": "#2166ac", "SAC": "#b2182b"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, b, a in METHODS:
        iqm, lo, hi, nmax = [], [], [], 0
        for g in grid:
            r = iqm_iqr([windowed_err(DATA[(lab, s)], g) for s in SEEDS if DATA[(lab, s)] is not None])
            if r is None:
                iqm.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                iqm.append(r[0]); lo.append(r[1]); hi.append(r[2]); nmax = max(nmax, r[3])
        iqm, lo, hi = map(np.array, (iqm, lo, hi))
        ax.plot(grid, iqm, color=colors[lab], label=f"{lab} (n≤{nmax})", lw=2)
        ax.fill_between(grid, lo, hi, color=colors[lab], alpha=0.15)
    ax.axhline(CONV_CM, color="grey", ls="--", lw=0.8); ax.text(1500, CONV_CM + 0.5, "conv 15cm", fontsize=8, color="grey")
    ax.axhline(GOAL_CM, color="grey", ls=":", lw=0.8); ax.text(1500, GOAL_CM + 0.5, "goal 5cm", fontsize=8, color="grey")
    ax.axvline(20000, color="k", ls=":", lw=0.6); ax.text(20200, ax.get_ylim()[1]*0.92, "learning_starts", fontsize=8)
    ax.set_xlabel("environment steps"); ax.set_ylabel("tracking error (cm), IQM ± IQR")
    ax.set_title("FetchSlide @250-290ms: delay-correction methods (8 seeds)")
    ax.set_ylim(0, 70)
    ax.legend(); fig.tight_layout(); fig.savefig("absp80k_training_curves.png", dpi=130)
    print("\n[plot] wrote absp80k_training_curves.png")
