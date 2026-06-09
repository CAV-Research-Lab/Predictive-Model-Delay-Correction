"""IQM (line) + IQR (shaded) training curves for PMDC vs A-SAC vs SAC at one delay.

Usage: python3 plot_iqm_curves.py <delay_label> <pmdc_run_dir> <out.png>
  e.g. python3 plot_iqm_curves.py 250-290ms runs_FetchSlide_sac_g95_tent1 plots/iqm_250-290.png
A-SAC and SAC are read from the multi-seed study at runs_FetchSlide/.
"""
import sys
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

delay = sys.argv[1]
pmdc_dir = sys.argv[2]
out = sys.argv[3]
NBINS = 50


def smooth(y, w=5):
    if w <= 1 or len(y) < w:
        return y
    yp = np.pad(y, w // 2, mode="edge")
    return np.convolve(yp, np.ones(w) / w, mode="valid")[:len(y)]


methods = {
    "PMDC (best cfg)": (f"{pmdc_dir}/pred_logs/PMDC/5/{delay}", "C0"),
    "A-SAC": (f"runs_FetchSlide/pred_logs/A-SAC/5/{delay}", "C1"),
    "SAC": (f"runs_FetchSlide/pred_logs/SAC/5/{delay}", "C2"),
}


def seed_curve(p):
    ev = sorted(Path(p).rglob("events.out.*"))
    if not ev:
        return None
    ea = event_accumulator.EventAccumulator(str(ev[-1]))
    ea.Reload()
    if "track/step_reward" not in ea.Tags().get("scalars", []):
        return None
    v = np.array([x.value for x in ea.Scalars("track/step_reward")])
    err = -v * 100.0  # cm
    return smooth(np.array([b.mean() for b in np.array_split(err, NBINS)]), 5)


def iqm_axis(curves):  # curves: (n_seeds, NBINS) -> per-bin IQM
    out = np.zeros(curves.shape[1])
    for i in range(curves.shape[1]):
        col = np.sort(curves[:, i]); n = len(col); k = int(np.floor(n * 0.25))
        out[i] = col[k:n - k].mean() if n - 2 * k > 0 else col.mean()
    return out


x = np.linspace(0, 60000, NBINS)
plt.figure(figsize=(7, 4.5))
summary = []
for name, (base, color) in methods.items():
    curves = [seed_curve(f"{base}/seed_{s}") for s in range(5)]
    curves = np.array([c for c in curves if c is not None])
    if not len(curves):
        continue
    iqm = iqm_axis(curves)
    lo, hi = np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0)
    plt.plot(x, iqm, label=name, color=color, lw=2)
    plt.fill_between(x, lo, hi, alpha=0.18, color=color)
    summary.append(f"{name}: final IQM={iqm[-1]:.1f}cm")
plt.xlabel("environment steps")
plt.ylabel("tracking error (cm)")
plt.title(f"FetchSlide {delay}: IQM (line) + IQR (shaded), 5 seeds")
plt.legend()
plt.ylim(0, None)
plt.grid(alpha=0.3)
plt.tight_layout()
Path(out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"saved {out}  |  " + "  ".join(summary))
