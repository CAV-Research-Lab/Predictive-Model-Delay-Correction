"""
Plot training curves comparing PMDC, A-SAC, and SAC under different time delays.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def smooth(y, window=51):
    """Gaussian-weighted moving average."""
    if len(y) < window:
        return y
    kernel = np.exp(-0.5 * (np.arange(window) - window // 2) ** 2 / (window / 6) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y, window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')


COLORS = {
    "PMDC": "#1f77b4",
    "A-SAC": "#ff7f0e",
    "SAC": "#2ca02c",
}

LABELS = {
    "PMDC": "PMDC (ours)",
    "A-SAC": "A-SAC",
    "SAC": "SAC",
}


def load_csv(results_dir, algorithm, n_models, delay_label, seed):
    path = Path(results_dir) / algorithm / str(n_models) / delay_label / f"seed_{seed}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["delayed_reward"])
    return df


def plot_comparison(results_dir="pred_results", n_models=5, delay_labels=None, seeds=None, output="PMDC_training_curves_csv.png"):
    if delay_labels is None:
        delay_labels = ["90-120ms", "250-290ms"]
    if seeds is None:
        seeds = [0]

    algorithms = ["PMDC", "A-SAC", "SAC"]
    n_delays = len(delay_labels)

    fig, axes = plt.subplots(1, n_delays, figsize=(6 * n_delays, 5), squeeze=False)

    for col, delay_label in enumerate(delay_labels):
        ax = axes[0][col]
        ax.set_title(f"Delay: {delay_label}", fontsize=14, fontweight='bold')

        for alg in algorithms:
            dfs = []
            for seed in seeds:
                df = load_csv(results_dir, alg, n_models, delay_label, seed)
                if df is not None:
                    dfs.append(df)

            if not dfs:
                print(f"  No data for {alg} {delay_label}")
                continue

            # Align on step and average across seeds
            all_rewards = []
            max_step = max(df["step"].max() for df in dfs)
            common_steps = np.linspace(0, max_step, 800)

            for df in dfs:
                interpolated = np.interp(common_steps, df["step"].values, df["delayed_reward"].values)
                all_rewards.append(interpolated)

            mean_reward = np.mean(all_rewards, axis=0)
            std_reward = np.std(all_rewards, axis=0) if len(all_rewards) > 1 else np.zeros_like(mean_reward)

            smoothed_mean = smooth(mean_reward)
            smoothed_std = smooth(std_reward)

            color = COLORS[alg]
            ax.plot(common_steps, smoothed_mean, label=LABELS[alg], color=color, linewidth=2)
            if len(seeds) > 1:
                ax.fill_between(
                    common_steps,
                    smoothed_mean - smoothed_std,
                    smoothed_mean + smoothed_std,
                    alpha=0.2, color=color
                )

            # Report final performance
            final_window = smoothed_mean[-100:]
            print(f"  {alg:6s} | {delay_label} | mean_last_100: {np.mean(final_window):.4f} ± {np.std(final_window):.4f}")

        ax.set_xlabel("Training steps", fontsize=12)
        ax.set_ylabel("Delayed reward (per step)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_step if 'max_step' in dir() else 80000)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output}")
    plt.close()


def print_summary_table(results_dir="pred_results", n_models=5, delay_labels=None, seeds=None):
    if delay_labels is None:
        delay_labels = ["90-120ms", "250-290ms"]
    if seeds is None:
        seeds = [0]

    algorithms = ["PMDC", "A-SAC", "SAC"]

    print("\n=== Final Performance Summary ===")
    print(f"{'Algorithm':<12} {'Delay':<14} {'Mean last-100':<20} {'Best rolling-20':<20}")
    print("-" * 70)

    for delay_label in delay_labels:
        for alg in algorithms:
            values = []
            for seed in seeds:
                df = load_csv(results_dir, alg, n_models, delay_label, seed)
                if df is not None:
                    rewards = df["delayed_reward"].dropna().values
                    if len(rewards) >= 100:
                        mean_last = np.mean(rewards[-100:])
                    else:
                        mean_last = np.mean(rewards)
                    # Best rolling 20 window
                    if len(rewards) >= 20:
                        rolling = np.array([np.mean(rewards[i:i+20]) for i in range(len(rewards)-19)])
                        best_rolling = np.max(rolling)
                    else:
                        best_rolling = np.max(rewards) if len(rewards) > 0 else float('nan')
                    values.append((mean_last, best_rolling))

            if values:
                means = [v[0] for v in values]
                bests = [v[1] for v in values]
                print(f"{alg:<12} {delay_label:<14} {np.mean(means):.4f} ± {np.std(means):.4f}    {np.mean(bests):.4f} ± {np.std(bests):.4f}")
            else:
                print(f"{alg:<12} {delay_label:<14} (no data)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PMDC training curves")
    parser.add_argument("--results-dir", default="pred_results")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--delay-labels", nargs="+", default=["90-120ms", "250-290ms"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output", default="PMDC_training_curves_csv.png")
    args = parser.parse_args()

    print(f"Plotting results from: {args.results_dir}")
    print(f"Algorithms: PMDC, A-SAC, SAC")
    print(f"Delays: {args.delay_labels}")
    print(f"Seeds: {args.seeds}")
    print()

    print_summary_table(args.results_dir, args.n_models, args.delay_labels, args.seeds)
    plot_comparison(args.results_dir, args.n_models, args.delay_labels, args.seeds, args.output)
