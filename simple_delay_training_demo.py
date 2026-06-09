"""Training-curve diagnostic for the simple delayed point-reaching task.

This complements ``simple_delay_type_demo.py``.  It uses the same environment
and random-delay wrappers, but trains SAC policies and evaluates them at regular
intervals so convergence can be inspected rather than only final performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from simple_delay_type_demo import (
    COLORS,
    VARIANT_ORDER,
    VARIANTS,
    DelaySpec,
    configure_style,
    iqm,
    make_delay_env,
)


EXPERIMENTS = {
    "constant": "Constant delay",
    "stochastic": "Stochastic observation delay",
}

CONDITION_COLORS = {
    "constant_10_10": "#2f5f8f",
    "constant_15_15": "#7f5aa2",
    "constant_20_20": "#b14e2c",
    "stochastic_0_10": "#3f7f4f",
    "stochastic_0_20": "#9a6b2f",
}


@dataclass(frozen=True)
class TrainingCondition:
    condition_id: str
    condition_label: str
    family: str
    spec: DelaySpec


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_args(args):
    return SimpleNamespace(
        horizon=args.horizon,
        action_scale=args.action_scale,
        process_noise=args.process_noise,
        goal_noise=args.goal_noise,
        success_threshold=args.success_threshold,
    )


def delay_spec(args, experiment, value=None):
    if experiment == "constant":
        delay = args.constant_delay if value is None else value
        return DelaySpec(
            kind="training_constant",
            act_min=delay,
            act_max=delay,
            obs_min=delay,
            obs_max=delay,
        )
    if experiment == "stochastic":
        upper = args.stochastic_upper if value is None else value
        return DelaySpec(
            kind="training_stochastic",
            act_min=0,
            act_max=0,
            obs_min=0,
            obs_max=upper,
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def build_requested_conditions(args):
    conditions = []
    for delay in sorted(set(args.constant_delays)):
        spec = delay_spec(args, "constant", delay)
        conditions.append(TrainingCondition(f"constant_{delay}_{delay}", spec.label, "constant", spec))
    for upper in sorted(set(args.stochastic_uppers)):
        spec = delay_spec(args, "stochastic", upper)
        conditions.append(TrainingCondition(f"stochastic_0_{upper}", spec.label, "stochastic", spec))
    return conditions


def build_legacy_conditions(args):
    conditions = []
    if "constant" in args.experiments:
        spec = delay_spec(args, "constant")
        conditions.append(
            TrainingCondition(f"constant_{args.constant_delay}_{args.constant_delay}", spec.label, "constant", spec)
        )
    if "stochastic" in args.experiments:
        spec = delay_spec(args, "stochastic")
        conditions.append(TrainingCondition(f"stochastic_0_{args.stochastic_upper}", spec.label, "stochastic", spec))
    return conditions


def build_training_matrix(args):
    if args.preset == "requested":
        conditions = {condition.condition_id: condition for condition in build_requested_conditions(args)}
        jobs = set()
        for delay in args.augmented_delay_magnitude:
            jobs.add((f"constant_{delay}_{delay}", args.augmented_variant))
        for variant in args.variants:
            jobs.add(("constant_10_10", variant))
        for condition_id in ["stochastic_0_10", "stochastic_0_20", "constant_10_10", "constant_20_20"]:
            for variant in args.variants:
                jobs.add((condition_id, variant))
        return conditions, sorted(jobs)

    conditions = {condition.condition_id: condition for condition in build_legacy_conditions(args)}
    jobs = [(condition_id, variant) for condition_id in conditions for variant in args.variants]
    return conditions, jobs


def build_env(args, spec, variant, seed):
    env = make_delay_env(spec, variant, seed, env_args(args))
    env.action_space.seed(seed)
    return env


def evaluate_model(model, args, condition, variant, seed, train_steps):
    spec = condition.spec
    env = build_env(args, spec, variant, seed)
    rows = []
    for episode in range(args.n_eval_episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        delayed_rewards = []
        true_rewards = []
        distances = []
        last_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            distance = float(env.unwrapped._distance())
            delayed_rewards.append(float(reward))
            true_rewards.append(-distance)
            distances.append(distance)

        rows.append(
            {
                "experiment": condition.family,
                "experiment_label": EXPERIMENTS.get(condition.family, condition.family),
                "condition_id": condition.condition_id,
                "condition_label": condition.condition_label,
                "delay_family": condition.family,
                "variant": variant,
                "variant_label": VARIANTS[variant],
                "delay_label": spec.label,
                "act_min": spec.act_min,
                "act_max": spec.act_max,
                "obs_min": spec.obs_min,
                "obs_max": spec.obs_max,
                "seed": seed,
                "train_steps": train_steps,
                "episode": episode,
                "episode_steps": len(true_rewards),
                "true_return": float(np.sum(true_rewards)),
                "delayed_return": float(np.sum(delayed_rewards)),
                "mean_distance": float(np.mean(distances)),
                "final_distance": float(distances[-1]),
                "min_distance": float(np.min(distances)),
                "success": float(last_info.get("is_success", distances[-1] <= args.success_threshold)),
            }
        )
    env.close()
    return rows


def train_one(args, condition, variant, seed):
    seed_everything(seed)
    spec = condition.spec
    env = build_env(args, spec, variant, seed)
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=min(args.learning_starts, max(1, args.steps - 1)),
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=args.verbose,
    )

    rows = []
    for train_steps in range(args.eval_every, args.steps + 1, args.eval_every):
        model.learn(
            total_timesteps=args.eval_every,
            reset_num_timesteps=(train_steps == args.eval_every),
            log_interval=args.log_interval,
            progress_bar=False,
        )
        eval_seed = args.eval_seed + seed * 10_000 + train_steps
        eval_rows = evaluate_model(model, args, condition, variant, eval_seed, train_steps)
        rows.extend(eval_rows)
        returns = [row["true_return"] for row in eval_rows]
        successes = [row["success"] for row in eval_rows]
        print(
            f"{condition.condition_id:18s} {variant:24s} seed={seed} "
            f"step={train_steps:6d} "
            f"return={np.mean(returns):8.2f} success={np.mean(successes):.2f}",
            flush=True,
        )

    env.close()
    return rows


def run_training(args):
    conditions, jobs = build_training_matrix(args)
    rows = []
    for condition_id, variant in jobs:
        condition = conditions[condition_id]
        for seed in args.seeds:
            rows.extend(train_one(args, condition, variant, seed))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, group_cols, metric):
    buckets = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        buckets.setdefault(key, []).append(float(row[metric]))
    out = []
    for key, values in buckets.items():
        values = np.asarray(values, dtype=float)
        q1, q3 = np.quantile(values, [0.25, 0.75])
        item = dict(zip(group_cols, key))
        item.update({"iqm": iqm(values), "q1": float(q1), "q3": float(q3), "n": int(len(values))})
        out.append(item)
    return out


def write_summary(path, rows):
    group_cols = [
        "experiment",
        "experiment_label",
        "condition_id",
        "condition_label",
        "delay_family",
        "variant",
        "variant_label",
        "delay_label",
        "train_steps",
    ]
    summary = []
    for metric in ["true_return", "mean_distance", "success"]:
        for row in aggregate(rows, group_cols, metric):
            row = dict(row)
            row["metric"] = metric
            summary.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = group_cols + ["metric", "iqm", "q1", "q3", "n"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def plot_metric(rows, metric, ylabel, output_path):
    configure_style()
    experiments = [experiment for experiment in EXPERIMENTS if experiment in {row["experiment"] for row in rows}]
    fig, axes = plt.subplots(
        1,
        len(experiments),
        figsize=(5.3 * len(experiments), 4.0),
        sharey=len(experiments) > 1,
        constrained_layout=True,
    )
    if len(experiments) == 1:
        axes = [axes]

    summary = aggregate(
        rows,
        ["experiment", "experiment_label", "variant", "variant_label", "delay_label", "train_steps"],
        metric,
    )
    for ax, experiment in zip(axes, experiments):
        exp_rows = [row for row in summary if row["experiment"] == experiment]
        delay_label = exp_rows[0]["delay_label"] if exp_rows else ""
        for variant in VARIANT_ORDER:
            part = sorted([row for row in exp_rows if row["variant"] == variant], key=lambda item: item["train_steps"])
            if not part:
                continue
            xs = np.asarray([row["train_steps"] for row in part], dtype=float)
            ys = np.asarray([row["iqm"] for row in part], dtype=float)
            q1 = np.asarray([row["q1"] for row in part], dtype=float)
            q3 = np.asarray([row["q3"] for row in part], dtype=float)
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=COLORS[variant], label=VARIANTS[variant])
            ax.fill_between(xs, q1, q3, color=COLORS[variant], alpha=0.16, linewidth=0)
        ax.set_title(f"{EXPERIMENTS[experiment]} {delay_label}")
        ax.set_xlabel("Training environment steps")
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_lines(rows, series, metric, ylabel, title, output_path, color_by):
    configure_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    group_cols = [
        "condition_id",
        "condition_label",
        "variant",
        "variant_label",
        "train_steps",
    ]
    summary = aggregate(rows, group_cols, metric)
    for item in series:
        part = sorted(
            [
                row
                for row in summary
                if row["condition_id"] == item["condition_id"] and row["variant"] == item["variant"]
            ],
            key=lambda row: row["train_steps"],
        )
        if not part:
            continue
        xs = np.asarray([row["train_steps"] for row in part], dtype=float)
        ys = np.asarray([row["iqm"] for row in part], dtype=float)
        q1 = np.asarray([row["q1"] for row in part], dtype=float)
        q3 = np.asarray([row["q3"] for row in part], dtype=float)
        if color_by == "variant":
            color = COLORS[item["variant"]]
        else:
            color = CONDITION_COLORS.get(item["condition_id"], "#444444")
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=item["label"])
        ax.fill_between(xs, q1, q3, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_panel_comparison(rows, metric, ylabel, title, output_path):
    configure_style()
    fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(14.0, 4.1), sharey=True, constrained_layout=True)
    group_cols = [
        "condition_id",
        "condition_label",
        "variant",
        "variant_label",
        "train_steps",
    ]
    summary = aggregate(rows, group_cols, metric)
    condition_ids = ["constant_10_10", "constant_20_20", "stochastic_0_10", "stochastic_0_20"]
    handles = []
    labels = []
    for ax, variant in zip(axes, VARIANT_ORDER):
        for condition_id in condition_ids:
            part = sorted(
                [
                    row
                    for row in summary
                    if row["condition_id"] == condition_id and row["variant"] == variant
                ],
                key=lambda row: row["train_steps"],
            )
            if not part:
                continue
            xs = np.asarray([row["train_steps"] for row in part], dtype=float)
            ys = np.asarray([row["iqm"] for row in part], dtype=float)
            q1 = np.asarray([row["q1"] for row in part], dtype=float)
            q3 = np.asarray([row["q3"] for row in part], dtype=float)
            label = part[0]["condition_label"]
            color = CONDITION_COLORS.get(condition_id, "#444444")
            line = ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=label)[0]
            ax.fill_between(xs, q1, q3, color=color, alpha=0.16, linewidth=0)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(VARIANTS[variant])
        ax.set_xlabel("Training environment steps")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_requested_comparisons(output_dir, rows, metric, prefix, ylabel, augmented_variant):
    augmented_series = [
        {
            "condition_id": f"constant_{delay}_{delay}",
            "variant": augmented_variant,
            "label": f"Augmented {delay},{delay}",
        }
        for delay in [10, 15, 20]
    ]
    plot_comparison_lines(
        rows,
        augmented_series,
        metric,
        ylabel,
        "Augmented-State Convergence as Constant Delay Increases",
        output_dir / f"{prefix}_augmented_delay_magnitude.png",
        color_by="condition",
    )

    state_series = [
        {
            "condition_id": "constant_10_10",
            "variant": variant,
            "label": VARIANTS[variant],
        }
        for variant in VARIANT_ORDER
    ]
    plot_comparison_lines(
        rows,
        state_series,
        metric,
        ylabel,
        "State Information Convergence at Constant Delay (10,10)",
        output_dir / f"{prefix}_state_information_10_10.png",
        color_by="variant",
    )

    plot_panel_comparison(
        rows,
        metric,
        ylabel,
        "Constant vs Stochastic Delay Convergence",
        output_dir / f"{prefix}_constant_vs_stochastic.png",
    )


def plot_results(output_dir, rows, augmented_variant="augmented_action_delay"):
    if {row["condition_id"] for row in rows}.issuperset(
        {"constant_10_10", "constant_15_15", "constant_20_20", "stochastic_0_10", "stochastic_0_20"}
    ):
        plot_requested_comparisons(
            output_dir,
            rows,
            "true_return",
            "reward",
            "IQM true evaluation return",
            augmented_variant,
        )
        plot_requested_comparisons(
            output_dir,
            rows,
            "mean_distance",
            "distance",
            "IQM mean goal distance",
            augmented_variant,
        )
    else:
        plot_metric(rows, "true_return", "IQM true evaluation return", output_dir / "reward_convergence.png")
        plot_metric(rows, "mean_distance", "IQM mean goal distance", output_dir / "distance_convergence.png")
        plot_metric(rows, "success", "IQM final-step success rate", output_dir / "success_convergence.png")


def write_process_doc(path, args):
    payload = {
        "environment": "PointReachEnv",
        "trainer": "Stable-Baselines3 SAC",
        "delay_wrapper": "wrappers_rd.RandomDelayWrapper via unseen/action-buffer/action-buffer-plus-delay variants",
        "reward_curve_metric": "IQM true evaluation return with IQR bands",
        "distance_curve_metric": "IQM mean goal distance with IQR bands",
        "experiments": args.experiments,
        "variants": args.variants,
        "preset": args.preset,
        "constant_delays": args.constant_delays,
        "stochastic_uppers": args.stochastic_uppers,
        "augmented_delay_magnitude_variant": args.augmented_variant,
        "seeds": args.seeds,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "n_eval_episodes": args.n_eval_episodes,
        "constant_delay": [args.constant_delay, args.constant_delay],
        "stochastic_delay": [0, f"0-{args.stochastic_upper}"],
        "horizon": args.horizon,
        "action_scale": args.action_scale,
        "process_noise": args.process_noise,
        "goal_noise": args.goal_noise,
        "success_threshold": args.success_threshold,
        "assumption": "This is a fast convergence diagnostic for the delay mechanism, not a replacement for full Fetch training.",
    }
    if args.preset == "requested":
        output_lines = [
            "- `reward_augmented_delay_magnitude.png`: augmented-state reward convergence for `(10,10)`, `(15,15)`, and `(20,20)`.",
            "- `reward_state_information_10_10.png`: unseen vs augmented reward convergence at `(10,10)`.",
            "- `reward_constant_vs_stochastic.png`: constant-vs-stochastic reward convergence by method.",
            "- `distance_augmented_delay_magnitude.png`: matching true-distance version for the delay-magnitude comparison.",
            "- `distance_state_information_10_10.png`: matching true-distance version for the state-information comparison.",
            "- `distance_constant_vs_stochastic.png`: matching true-distance version for the constant-vs-stochastic comparison.",
        ]
    else:
        output_lines = [
            "- `reward_convergence.png`: reward convergence over training steps.",
            "- `distance_convergence.png`: distance convergence over training steps.",
            "- `success_convergence.png`: success-rate convergence over training steps.",
        ]

    text = [
        "# Simple Delay Training-Curve Demonstration",
        "",
        "This run trains SAC on the same delayed point-reaching setup used by `simple_delay_type_demo.py`.",
        "The curves evaluate the current policy at regular training intervals and aggregate evaluation episodes/seeds with IQM/IQR.",
        "",
        "## What The Curves Mean",
        "",
        "- `reward_convergence.png` plots true base-environment evaluation return, so higher is better.",
        "- `distance_convergence.png` plots true mean goal distance, so lower is better.",
        "- `success_convergence.png` plots final-step success rate.",
        "- The learner still receives the delayed reward emitted by the random-delay wrapper during training.",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "## Outputs",
        "",
        "- `results.csv`: per-evaluation-episode rows.",
        "- `summary.csv`: IQM/IQR summaries for reward, distance, and success.",
        *output_lines,
    ]
    path.write_text("\n".join(text) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="diagnostics/simple_delay_training_demo")
    parser.add_argument("--preset", choices=["requested", "legacy"], default="requested")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=["constant", "stochastic"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANT_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--n-eval-episodes", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=500_000)
    parser.add_argument("--constant-delay", type=int, default=10)
    parser.add_argument("--stochastic-upper", type=int, default=20)
    parser.add_argument("--constant-delays", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--stochastic-uppers", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--augmented-delay-magnitude", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--augmented-variant", choices=VARIANTS, default="augmented_action_delay")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--process-noise", type=float, default=0.005)
    parser.add_argument("--goal-noise", type=float, default=0.08)
    parser.add_argument("--success-threshold", type=float, default=0.06)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--net-arch", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    output_dir = Path(args.output_dir)
    rows = run_training(args)
    write_csv(output_dir / "results.csv", rows)
    write_summary(output_dir / "summary.csv", rows)
    plot_results(output_dir, rows, args.augmented_variant)
    write_process_doc(output_dir / "PROCESS.md", args)
    print(f"Saved results: {output_dir / 'results.csv'}")
    print(f"Saved summary: {output_dir / 'summary.csv'}")
    if args.preset == "requested":
        print(
            "Saved reward plots: "
            f"{output_dir / 'reward_augmented_delay_magnitude.png'}, "
            f"{output_dir / 'reward_state_information_10_10.png'}, "
            f"{output_dir / 'reward_constant_vs_stochastic.png'}"
        )
    else:
        print(
            "Saved plots: "
            f"{output_dir / 'reward_convergence.png'}, "
            f"{output_dir / 'distance_convergence.png'}, "
            f"{output_dir / 'success_convergence.png'}"
        )
    print(f"Saved process doc: {output_dir / 'PROCESS.md'}")


if __name__ == "__main__":
    main()
