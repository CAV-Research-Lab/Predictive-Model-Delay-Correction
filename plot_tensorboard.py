"""
Plot training curves from TensorBoard logs.
Compares PMDC, A-SAC, SAC using track/step_reward (the consistent true-tracking metric).
This is the AUTHORITATIVE plotter (see CLAUDE.md). For non-FetchPush envs, point --log-base
at the isolated tree, e.g. --log-base runs_FetchPickAndPlace/pred_logs.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


COLORS = {"PMDC": "#1f77b4", "A-SAC": "#ff7f0e", "SAC": "#2ca02c"}
LABELS = {"PMDC": "PMDC (ours)", "A-SAC": "A-SAC", "SAC": "SAC"}


def smooth(arr, window=50):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def load_tb_data(logdir, tag="track/step_reward"):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def plot_from_tensorboard(log_base="pred_logs", delay_labels=None, n_models=5,
                          output="PMDC_training_curves.png", title=None):
    if delay_labels is None:
        delay_labels = ["250-290ms", "90-120ms"]

    available_delays = []
    for d in delay_labels:
        for alg in ["PMDC", "A-SAC", "SAC"]:
            logdir = f"{log_base}/{alg}/{n_models}/{d}/seed_0"
            if Path(logdir).exists() and list(Path(logdir).iterdir()):
                available_delays.append(d)
                break

    if not available_delays:
        print("No tensorboard logs found!")
        return

    n_plots = len(available_delays)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), squeeze=False)
    header = title or "PMDC vs A-SAC vs SAC"
    fig.suptitle(
        f"{header}: Tracking Performance During Training\n"
        "(metric: actual tracking distance from REnvPDNormObs — consistent for all algorithms)",
        fontsize=11, y=1.02,
    )

    for col, delay in enumerate(available_delays):
        ax = axes[0][col]
        ax.set_title(f"Delay: {delay}", fontsize=13, fontweight="bold")
        max_step = 60000

        for alg in ["PMDC", "A-SAC", "SAC"]:
            root = Path(f"{log_base}/{alg}/{n_models}/{delay}/seed_0")
            if not root.exists():
                continue
            subdirs = sorted(root.iterdir())
            if not subdirs:
                continue
            steps, values = load_tb_data(str(subdirs[-1]))
            if steps is None:
                print(f"  No data for {alg} {delay}")
                continue

            max_step = max(max_step, int(steps[-1]))
            ax.plot(steps, smooth(values), label=LABELS[alg], color=COLORS[alg], linewidth=2)
            band = smooth(values, window=20)
            ax.fill_between(steps, band - 0.05, band + 0.05, alpha=0.1, color=COLORS[alg])

            mean100 = np.mean(values[-100:]) if len(values) >= 100 else float(np.mean(values))
            print(f"  {alg:6s} {delay}: step={steps[-1]}, mean_last100={mean100:.4f}")

        ax.set_xlabel("Training steps", fontsize=11)
        ax.set_ylabel("Tracking reward (per step)\n[closer to 0 = better tracking]", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.0, 0.1)
        ax.axvline(x=20000, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(20200, -0.95, "SAC\ntraining\nstarts", fontsize=8, color="gray", alpha=0.8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {output}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-base", default="pred_logs")
    parser.add_argument("--delays", nargs="+", default=["250-290ms", "90-120ms"])
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--output", default="PMDC_training_curves.png")
    parser.add_argument("--title", default=None, help="Header text, e.g. an environment name.")
    args = parser.parse_args()

    print(f"Plotting from: {args.log_base}\nDelays: {args.delays}\n")
    plot_from_tensorboard(args.log_base, args.delays, args.n_models, args.output, args.title)
