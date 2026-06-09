"""Fast controller-training curves for delayed point reaching.

This is a lightweight policy-optimisation diagnostic.  It uses the same
``PointReachEnv`` and random-delay wrappers as ``simple_delay_type_demo.py``,
but trains a small proportional controller with cross-entropy search.  The
purpose is to isolate the delay mechanism and produce clean convergence curves
without requiring a long SAC run.
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

from simple_delay_type_demo import (
    COLORS,
    VARIANT_ORDER,
    VARIANTS,
    DelaySpec,
    configure_style,
    iqm,
    make_delay_env,
    split_augmented_obs,
)


CONDITION_COLORS = {
    "constant_10_10": "#2f5f8f",
    "constant_15_15": "#7f5aa2",
    "constant_20_20": "#b14e2c",
    "stochastic_0_0": "#2f5f8f",
    "stochastic_0_10": "#3f7f4f",
    "stochastic_0_20": "#9a6b2f",
}
LINE_STYLES = {
    "unseen": "-",
    "augmented_action": "-",
    "augmented_action_delay": "-",
}
MARKERS = {
    "unseen": "o",
    "augmented_action": "s",
    "augmented_action_delay": "^",
}


@dataclass(frozen=True)
class TrainingCondition:
    condition_id: str
    condition_label: str
    family: str
    spec: DelaySpec


def env_args(args):
    return SimpleNamespace(
        horizon=args.horizon,
        action_scale=args.action_scale,
        process_noise=args.process_noise,
        goal_noise=args.goal_noise,
        success_threshold=args.success_threshold,
    )


def constant_condition(delay):
    spec = DelaySpec("constant", delay, delay, delay, delay)
    return TrainingCondition(f"constant_{delay}_{delay}", spec.label, "constant", spec)


def stochastic_condition(upper, mode):
    if mode == "both":
        spec = DelaySpec("stochastic", 0, upper, 0, upper)
        label = "(0-0,0-0)" if upper == 0 else f"(0-{upper},0-{upper})"
        return TrainingCondition(f"stochastic_0_{upper}", label, "stochastic", spec)
    spec = DelaySpec("stochastic", 0, 0, 0, upper)
    label = "(0,0-0)" if upper == 0 else f"(0,0-{upper})"
    return TrainingCondition(f"stochastic_0_{upper}", label, "stochastic", spec)


def build_conditions(args):
    conditions = {}
    for delay in sorted(set(args.constant_delays)):
        condition = constant_condition(delay)
        conditions[condition.condition_id] = condition
    stochastic_uppers = set(args.stochastic_uppers)
    stochastic_uppers.add(args.stochastic_state_upper)
    for upper in sorted(stochastic_uppers):
        condition = stochastic_condition(upper, args.stochastic_mode)
        conditions[condition.condition_id] = condition
    return conditions


def build_training_matrix(args):
    conditions = build_conditions(args)
    if args.matrix == "stochastic_augmented":
        jobs = [(f"stochastic_0_{upper}", args.augmented_variant) for upper in sorted(set(args.stochastic_uppers))]
        return conditions, jobs
    if args.matrix == "stochastic_state_info":
        jobs = [(f"stochastic_0_{args.stochastic_state_upper}", variant) for variant in VARIANT_ORDER]
        return conditions, jobs

    jobs = set()
    for delay in args.augmented_delay_magnitude:
        jobs.add((f"constant_{delay}_{delay}", args.augmented_variant))
    for variant in args.variants:
        jobs.add(("constant_10_10", variant))
    for condition_id in ["stochastic_0_10", "stochastic_0_20", "constant_10_10", "constant_20_20"]:
        for variant in args.variants:
            jobs.add((condition_id, variant))
    return conditions, sorted(jobs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(theta):
    theta = np.asarray(theta, dtype=float)
    kp = 0.05 + 5.95 * sigmoid(theta[0])
    correction_scale = 2.0 * sigmoid(theta[1])
    return float(kp), float(correction_scale)


def estimate_position(obs, variant, spec, action_scale, correction_scale):
    if variant == "unseen":
        obs = np.asarray(obs, dtype=np.float32)
        return obs[:2], obs[2:4]

    include_delay_values = variant == "augmented_action_delay"
    base_obs, actions, delay_values = split_augmented_obs(obs, include_delay_values)
    delayed_position = base_obs[:2]
    goal = base_obs[2:4]

    if include_delay_values and delay_values is not None:
        alpha = int(round(float(delay_values[0])))
        beta = int(round(float(delay_values[2])))
        replay_len = max(0, min(len(actions), alpha + beta))
    else:
        replay_len = max(0, min(len(actions), spec.nominal_total_delay))

    if replay_len <= 0:
        correction = np.zeros(2, dtype=np.float32)
    else:
        correction = action_scale * np.sum(actions[1 : replay_len + 1], axis=0)
    return delayed_position + correction_scale * correction, goal


def controller_action(obs, variant, spec, args, theta):
    kp, correction_scale = decode(theta)
    position, goal = estimate_position(obs, variant, spec, args.action_scale, correction_scale)
    action = kp * (goal - position)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_episode(args, condition, variant, theta, seed):
    env = make_delay_env(condition.spec, variant, seed, env_args(args))
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    true_rewards = []
    delayed_rewards = []
    distances = []
    last_info = {}

    while not (terminated or truncated):
        action = controller_action(obs, variant, condition.spec, args, theta)
        obs, reward, terminated, truncated, last_info = env.step(action)
        distance = float(env.unwrapped._distance())
        true_rewards.append(-distance)
        delayed_rewards.append(float(reward))
        distances.append(distance)

    env.close()
    return {
        "episode_steps": len(true_rewards),
        "true_return": float(np.sum(true_rewards)),
        "delayed_return": float(np.sum(delayed_rewards)),
        "mean_distance": float(np.mean(distances)),
        "final_distance": float(distances[-1]),
        "min_distance": float(np.min(distances)),
        "success": float(last_info.get("is_success", distances[-1] <= args.success_threshold)),
    }


def score_candidate(args, condition, variant, theta, seed_base, n_episodes):
    returns = []
    for episode in range(n_episodes):
        metrics = run_episode(args, condition, variant, theta, seed_base + episode)
        returns.append(metrics["true_return"])
    return float(np.mean(returns))


def evaluate_policy(args, condition, variant, theta, seed_base, train_steps, iteration):
    rows = []
    kp, correction_scale = decode(theta)
    for episode in range(args.eval_episodes):
        metrics = run_episode(args, condition, variant, theta, seed_base + episode)
        rows.append(
            {
                "condition_id": condition.condition_id,
                "condition_label": condition.condition_label,
                "delay_family": condition.family,
                "variant": variant,
                "variant_label": VARIANTS[variant],
                "delay_label": condition.spec.label,
                "act_min": condition.spec.act_min,
                "act_max": condition.spec.act_max,
                "obs_min": condition.spec.obs_min,
                "obs_max": condition.spec.obs_max,
                "seed": seed_base,
                "iteration": iteration,
                "train_steps": train_steps,
                "episode": episode,
                "kp": kp,
                "correction_scale": correction_scale,
                **metrics,
            }
        )
    return rows


def train_one(args, condition, variant, seed):
    rng = np.random.default_rng(seed)
    mean = np.asarray(args.initial_mean, dtype=float)
    std = np.asarray(args.initial_std, dtype=float)
    best_theta = mean.copy()
    best_score = -np.inf
    rows = []
    train_steps = 0

    for iteration in range(args.iterations + 1):
        eval_seed = args.eval_seed + seed * 100_000 + iteration * 1_000
        rows.extend(evaluate_policy(args, condition, variant, best_theta, eval_seed, train_steps, iteration))
        if iteration == args.iterations:
            break

        candidates = rng.normal(mean, std, size=(args.population, len(mean)))
        scores = []
        for candidate_idx, theta in enumerate(candidates):
            train_seed = args.train_seed + seed * 1_000_000 + iteration * 10_000 + candidate_idx * 100
            score = score_candidate(args, condition, variant, theta, train_seed, args.train_episodes)
            scores.append(score)
            train_steps += args.train_episodes * args.horizon

        scores = np.asarray(scores, dtype=float)
        elite_idx = np.argsort(scores)[-args.elites :]
        elites = candidates[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), args.min_std)
        if float(np.max(scores)) > best_score:
            best_score = float(np.max(scores))
            best_theta = candidates[int(np.argmax(scores))].copy()

        print(
            f"{condition.condition_id:18s} {variant:24s} seed={seed} "
            f"iter={iteration + 1:02d} steps={train_steps:6d} "
            f"best_return={best_score:8.2f}",
            flush=True,
        )

    return rows


def run_training(args):
    rows = []
    conditions, jobs = build_training_matrix(args)
    for condition_id, variant in jobs:
        condition = conditions[condition_id]
        for seed in args.seeds:
            seed_everything(seed)
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


def filter_plot_rows(rows, max_plot_steps):
    if max_plot_steps is None:
        return rows
    return [row for row in rows if float(row["train_steps"]) <= max_plot_steps]


def line_marker(variant, show_markers):
    return MARKERS[variant] if show_markers else None


def write_summary(path, rows):
    group_cols = [
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


def plot_comparison_lines(rows, series, metric, ylabel, title, output_path, color_by, max_plot_steps, show_markers):
    configure_style()
    rows = filter_plot_rows(rows, max_plot_steps)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    group_cols = ["condition_id", "condition_label", "variant", "variant_label", "train_steps"]
    summary = aggregate(rows, group_cols, metric)
    for item in series:
        part = sorted(
            [
                row
                for row in summary
                if row["condition_id"] == item["condition_id"] and row["variant"] == item["variant"]
            ],
            key=lambda row: float(row["train_steps"]),
        )
        if not part:
            continue
        xs = np.asarray([row["train_steps"] for row in part], dtype=float)
        ys = np.asarray([row["iqm"] for row in part], dtype=float)
        q1 = np.asarray([row["q1"] for row in part], dtype=float)
        q3 = np.asarray([row["q3"] for row in part], dtype=float)
        color = COLORS[item["variant"]] if color_by == "variant" else CONDITION_COLORS[item["condition_id"]]
        ax.plot(
            xs,
            ys,
            marker=line_marker(item["variant"], show_markers),
            linestyle=LINE_STYLES[item["variant"]],
            linewidth=1.8,
            color=color,
            label=item["label"],
        )
        ax.fill_between(xs, q1, q3, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(ylabel)
    if max_plot_steps is not None:
        ax.set_xlim(0, max_plot_steps)
    ax.legend(loc="best", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_panel_comparison(rows, metric, ylabel, title, output_path, max_plot_steps, show_markers):
    configure_style()
    rows = filter_plot_rows(rows, max_plot_steps)
    fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(14.0, 4.1), sharey=True, constrained_layout=True)
    group_cols = ["condition_id", "condition_label", "variant", "variant_label", "train_steps"]
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
                key=lambda row: float(row["train_steps"]),
            )
            if not part:
                continue
            xs = np.asarray([row["train_steps"] for row in part], dtype=float)
            ys = np.asarray([row["iqm"] for row in part], dtype=float)
            q1 = np.asarray([row["q1"] for row in part], dtype=float)
            q3 = np.asarray([row["q3"] for row in part], dtype=float)
            label = part[0]["condition_label"]
            color = CONDITION_COLORS[condition_id]
            line = ax.plot(
                xs,
                ys,
                marker=line_marker(variant, show_markers),
                linestyle=LINE_STYLES[variant],
                linewidth=1.8,
                color=color,
                label=label,
            )[0]
            ax.fill_between(xs, q1, q3, color=color, alpha=0.16, linewidth=0)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(VARIANTS[variant])
        ax.set_xlabel("Training environment steps")
        if max_plot_steps is not None:
            ax.set_xlim(0, max_plot_steps)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_requested(output_dir, rows, metric, prefix, ylabel, augmented_variant, max_plot_steps, show_markers):
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
        "condition",
        max_plot_steps,
        show_markers,
    )

    state_series = [
        {"condition_id": "constant_10_10", "variant": variant, "label": VARIANTS[variant]}
        for variant in VARIANT_ORDER
    ]
    plot_comparison_lines(
        rows,
        state_series,
        metric,
        ylabel,
        "State Information Convergence at Constant Delay (10,10)",
        output_dir / f"{prefix}_state_information_10_10.png",
        "variant",
        max_plot_steps,
        show_markers,
    )

    plot_panel_comparison(
        rows,
        metric,
        ylabel,
        "Constant vs Stochastic Delay Convergence",
        output_dir / f"{prefix}_constant_vs_stochastic.png",
        max_plot_steps,
        show_markers,
    )


def plot_stochastic_augmented(output_dir, rows, metric, prefix, ylabel, augmented_variant, max_plot_steps, show_markers):
    stochastic_series = [
        {
            "condition_id": f"stochastic_0_{upper}",
            "variant": augmented_variant,
            "label": f"Augmented (0-{upper},0-{upper})",
        }
        for upper in [0, 10, 20]
    ]
    plot_comparison_lines(
        rows,
        stochastic_series,
        metric,
        ylabel,
        "Augmented-State Convergence as Stochastic Delay Range Increases",
        output_dir / f"{prefix}_augmented_stochastic_delay_magnitude.png",
        "condition",
        max_plot_steps,
        show_markers,
    )


def plot_stochastic_state_information(output_dir, rows, metric, prefix, ylabel, upper, max_plot_steps, show_markers):
    condition_id = f"stochastic_0_{upper}"
    state_series = [
        {"condition_id": condition_id, "variant": variant, "label": VARIANTS[variant]}
        for variant in VARIANT_ORDER
    ]
    plot_comparison_lines(
        rows,
        state_series,
        metric,
        ylabel,
        f"State Information Convergence at Stochastic Delay (0-{upper},0-{upper})",
        output_dir / f"{prefix}_state_information_stochastic_0_{upper}.png",
        "variant",
        max_plot_steps,
        show_markers,
    )


def plot_results(output_dir, rows, augmented_variant, max_plot_steps, show_markers):
    condition_ids = {row["condition_id"] for row in rows}
    variants_by_condition = {}
    for row in rows:
        variants_by_condition.setdefault(row["condition_id"], set()).add(row["variant"])
    if {"constant_10_10", "constant_15_15", "constant_20_20", "stochastic_0_10", "stochastic_0_20"}.issubset(
        condition_ids
    ):
        plot_requested(
            output_dir,
            rows,
            "true_return",
            "reward",
            "IQM true evaluation return",
            augmented_variant,
            max_plot_steps,
            show_markers,
        )
        plot_requested(
            output_dir,
            rows,
            "mean_distance",
            "distance",
            "IQM mean goal distance",
            augmented_variant,
            max_plot_steps,
            show_markers,
        )
        for condition_id in sorted(condition_ids):
            if not condition_id.startswith("stochastic_0_"):
                continue
            if not set(VARIANT_ORDER).issubset(variants_by_condition.get(condition_id, set())):
                continue
            upper = int(condition_id.rsplit("_", 1)[1])
            plot_stochastic_state_information(
                output_dir,
                rows,
                "true_return",
                "reward",
                "IQM true evaluation return",
                upper,
                max_plot_steps,
                show_markers,
            )
            plot_stochastic_state_information(
                output_dir,
                rows,
                "mean_distance",
                "distance",
                "IQM mean goal distance",
                upper,
                max_plot_steps,
                show_markers,
            )
    else:
        for condition_id in sorted(condition_ids):
            if not condition_id.startswith("stochastic_0_"):
                continue
            if not set(VARIANT_ORDER).issubset(variants_by_condition.get(condition_id, set())):
                continue
            upper = int(condition_id.rsplit("_", 1)[1])
            plot_stochastic_state_information(
                output_dir,
                rows,
                "true_return",
                "reward",
                "IQM true evaluation return",
                upper,
                max_plot_steps,
                show_markers,
            )
            plot_stochastic_state_information(
                output_dir,
                rows,
                "mean_distance",
                "distance",
                "IQM mean goal distance",
                upper,
                max_plot_steps,
                show_markers,
            )
    if {"stochastic_0_0", "stochastic_0_10", "stochastic_0_20"}.issubset(condition_ids):
        plot_stochastic_augmented(
            output_dir,
            rows,
            "true_return",
            "reward",
            "IQM true evaluation return",
            augmented_variant,
            max_plot_steps,
            show_markers,
        )
        plot_stochastic_augmented(
            output_dir,
            rows,
            "mean_distance",
            "distance",
            "IQM mean goal distance",
            augmented_variant,
            max_plot_steps,
            show_markers,
        )



def write_process_doc(path, args):
    payload = {
        "environment": "PointReachEnv",
        "trainer": "Cross-entropy optimisation of a proportional controller",
        "policy_parameters": "kp and action-history correction scale",
        "delay_wrapper": "wrappers_rd.RandomDelayWrapper via unseen/action-buffer/action-buffer-plus-delay variants",
        "reward_curve_metric": "IQM true evaluation return with IQR bands; higher is better",
        "distance_curve_metric": "IQM mean goal distance with IQR bands; lower is better",
        "constant_delays": args.constant_delays,
        "stochastic_uppers": args.stochastic_uppers,
        "stochastic_state_upper": args.stochastic_state_upper,
        "stochastic_mode": args.stochastic_mode,
        "matrix": args.matrix,
        "augmented_delay_magnitude_variant": args.augmented_variant,
        "variants": args.variants,
        "seeds": args.seeds,
        "iterations": args.iterations,
        "population": args.population,
        "elites": args.elites,
        "train_episodes_per_candidate": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "horizon": args.horizon,
        "assumption": "This is a fast delay-mechanism diagnostic, not a replacement for neural SAC policy training.",
    }
    text = [
        "# Simple Delay Controller-Training Demonstration",
        "",
        "This run optimises a small proportional reaching controller with cross-entropy search.",
        "It is intended to show convergence effects from delay and state information without requiring long SAC training.",
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
        "- `reward_augmented_delay_magnitude.png`: augmented-state reward convergence for `(10,10)`, `(15,15)`, and `(20,20)`.",
        "- `reward_augmented_stochastic_delay_magnitude.png`: augmented-state reward convergence for `(0-0,0-0)`, `(0-10,0-10)`, and `(0-20,0-20)`.",
        "- `reward_state_information_10_10.png`: unseen vs augmented reward convergence at `(10,10)`.",
        "- `reward_state_information_stochastic_0_20.png`: unseen vs augmented reward convergence at stochastic `(0-20,0-20)`.",
        "- `reward_constant_vs_stochastic.png`: constant-vs-stochastic reward convergence by method.",
        "- `distance_*`: matching true-distance versions of the requested reward plots.",
        "",
        "## Interpretation Notes",
        "",
        "- For constant delays, `+action buffer` and `+action buffer + delay values` are expected to overlap because the sampled delay is fixed and adds no extra runtime information.",
        "- The stochastic comparisons use stochastic action and observation ranges when `--stochastic-mode both` is selected.",
        "- A range such as `(0-20,0-20)` has lower mean delay than fixed `(20,20)`, so this plot tests widening stochastic ranges rather than proving every stochastic setting is harder than every constant setting.",
    ]
    path.write_text("\n".join(text) + "\n")


def write_plot_only_process_doc(path, args, source_path):
    payload = {
        "source_results_csv": str(source_path),
        "max_plot_steps": args.max_plot_steps,
        "show_markers": args.show_markers,
        "stochastic_mode": args.stochastic_mode,
        "note": "These are re-rendered plots from an existing evaluation-checkpoint CSV; no training was rerun.",
    }
    text = [
        "# Zoomed Delay Training-Curve Plots",
        "",
        "These plots were regenerated from an existing `results.csv` file.",
        "They cap the displayed x-axis and do not rerun training.",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "The curves are evaluation-checkpoint curves, not raw per-step training reward traces.",
    ]
    path.write_text("\n".join(text) + "\n")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="diagnostics/simple_delay_controller_training_demo")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--matrix", choices=["requested", "stochastic_augmented", "stochastic_state_info"], default="requested")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--max-plot-steps", type=float, default=None)
    parser.add_argument("--show-markers", action="store_true")
    parser.add_argument("--skip-process-doc", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANT_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--constant-delays", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--stochastic-uppers", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--stochastic-state-upper", type=int, default=20)
    parser.add_argument("--stochastic-mode", choices=["both", "observation"], default="both")
    parser.add_argument("--augmented-delay-magnitude", nargs="+", type=int, default=[10, 15, 20])
    parser.add_argument("--augmented-variant", choices=VARIANTS, default="augmented_action_delay")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=700_000)
    parser.add_argument("--eval-seed", type=int, default=900_000)
    parser.add_argument("--initial-mean", nargs="+", type=float, default=[-2.0, -2.0])
    parser.add_argument("--initial-std", nargs="+", type=float, default=[1.3, 1.3])
    parser.add_argument("--min-std", type=float, default=0.08)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--process-noise", type=float, default=0.005)
    parser.add_argument("--goal-noise", type=float, default=0.08)
    parser.add_argument("--success-threshold", type=float, default=0.06)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    source_path = Path(args.results_csv) if args.results_csv else output_dir / "results.csv"
    if args.plot_only:
        rows = read_csv(source_path)
    else:
        rows = run_training(args)
        write_csv(output_dir / "results.csv", rows)
        write_summary(output_dir / "summary.csv", rows)
    plot_results(output_dir, rows, args.augmented_variant, args.max_plot_steps, args.show_markers)
    if not args.skip_process_doc:
        if args.plot_only:
            write_plot_only_process_doc(output_dir / "PROCESS.md", args, source_path)
        else:
            write_process_doc(output_dir / "PROCESS.md", args)
    if args.plot_only:
        print(f"Loaded results: {source_path}")
    else:
        print(f"Saved results: {output_dir / 'results.csv'}")
        print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(
        "Saved reward plots: "
        f"{output_dir / 'reward_augmented_delay_magnitude.png'}, "
        f"{output_dir / 'reward_state_information_10_10.png'}, "
        f"{output_dir / 'reward_constant_vs_stochastic.png'}, "
        f"{output_dir / 'reward_augmented_stochastic_delay_magnitude.png'}"
    )
    if not args.skip_process_doc:
        print(f"Saved process doc: {output_dir / 'PROCESS.md'}")


if __name__ == "__main__":
    main()
