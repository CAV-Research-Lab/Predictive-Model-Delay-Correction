"""Fast delay-type demonstration on a simple point-reaching task.

This is intentionally not a replacement for the full Fetch experiments.  It is a
small, controlled setup that uses the same random-delay wrappers to illustrate
the qualitative delay story:

* unseen delayed state is worst,
* action-history state augmentation helps,
* explicit delay values help most under stochastic delays,
* larger constant delays and wider stochastic delay ranges are detrimental.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from wrappers_rd import AugmentedDelayInfoWrapper, AugmentedRandomDelayWrapper, UnseenRandomDelayWrapper


VARIANTS = {
    "unseen": "Unseen delay",
    "augmented_action": "Augmented (+action buffer)",
    "augmented_action_delay": "Augmented (+action buffer + delay values)",
}
VARIANT_ORDER = list(VARIANTS)
COLORS = {
    "unseen": "#2f5f8f",
    "augmented_action": "#b14e2c",
    "augmented_action_delay": "#3f7f4f",
}


@dataclass(frozen=True)
class DelaySpec:
    kind: str
    act_min: int
    act_max: int
    obs_min: int
    obs_max: int

    @property
    def act_range(self):
        return range(self.act_min, self.act_max + 1)

    @property
    def obs_range(self):
        return range(self.obs_min, self.obs_max + 1)

    @property
    def label(self):
        if self.obs_min == self.obs_max and self.act_min == self.act_max:
            return f"({self.act_max},{self.obs_max})"
        if self.act_min == self.act_max == 0:
            return f"(0,{self.obs_min}-{self.obs_max})"
        return f"({self.act_min}-{self.act_max},{self.obs_min}-{self.obs_max})"

    @property
    def x_value(self):
        return float(max(self.act_max, self.obs_max))

    @property
    def act_buffer_len(self):
        return self.act_max + 1

    @property
    def obs_buffer_len(self):
        return self.obs_max + 1

    @property
    def nominal_obs_delay(self):
        return int(round((self.obs_min + self.obs_max) / 2))

    @property
    def nominal_total_delay(self):
        return int(round((self.act_min + self.act_max + self.obs_min + self.obs_max) / 2))


class PointReachEnv(gym.Env):
    """A small 2D velocity-control reaching task with unobserved process noise."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon=80,
        action_scale=0.07,
        process_noise=0.005,
        goal_noise=0.08,
        success_threshold=0.06,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_scale = action_scale
        self.process_noise = process_noise
        self.goal_noise = goal_noise
        self.success_threshold = success_threshold
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.ones(2, dtype=np.float32)
        self.steps = 0

    def _obs(self):
        return np.concatenate([self.position, self.goal]).astype(np.float32)

    def _distance(self):
        return float(np.linalg.norm(self.position - self.goal))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.position = self.rng.uniform(-0.9, -0.6, size=2).astype(np.float32)
        self.goal = (np.asarray([0.75, 0.75]) + self.rng.normal(0.0, self.goal_noise, size=2)).astype(np.float32)
        return self._obs(), {"is_success": float(self._distance() <= self.success_threshold)}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        noise = self.rng.normal(0.0, self.process_noise, size=2).astype(np.float32)
        self.position = self.position + self.action_scale * action + noise
        self.steps += 1
        distance = self._distance()
        reward = -distance
        terminated = False
        truncated = self.steps >= self.horizon
        info = {
            "is_success": float(distance <= self.success_threshold),
            "goal_distance": distance,
            "position": self.position.copy(),
            "goal": self.goal.copy(),
        }
        return self._obs(), reward, terminated, truncated, info


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def iqm(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if len(values) < 4:
        return float(np.mean(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    middle = values[(values >= q1) & (values <= q3)]
    return float(np.mean(middle)) if len(middle) else float(np.mean(values))


def metric_label(metric):
    labels = {
        "final_distance": "IQM final goal distance",
        "mean_distance": "IQM mean goal distance",
        "min_distance": "IQM minimum goal distance",
    }
    return labels.get(metric, f"IQM {metric}")


def make_delay_env(spec, variant, seed, args):
    base_env = PointReachEnv(
        horizon=args.horizon,
        action_scale=args.action_scale,
        process_noise=args.process_noise,
        goal_noise=args.goal_noise,
        success_threshold=args.success_threshold,
    )
    base_env.action_space.seed(seed)
    kwargs = {
        "obs_delay_range": spec.obs_range,
        "act_delay_range": spec.act_range,
        "obs_buffer_len": spec.obs_buffer_len,
        "act_buffer_len": spec.act_buffer_len,
        "initial_action": np.zeros(2, dtype=np.float32),
    }
    if variant == "unseen":
        return UnseenRandomDelayWrapper(base_env, **kwargs)
    if variant == "augmented_action":
        return AugmentedRandomDelayWrapper(base_env, **kwargs)
    if variant == "augmented_action_delay":
        return AugmentedDelayInfoWrapper(base_env, **kwargs)
    raise ValueError(f"Unknown variant: {variant}")


def split_augmented_obs(obs, include_delay_values):
    obs = np.asarray(obs, dtype=np.float32)
    base_obs = obs[:4]
    if include_delay_values:
        action_values = obs[4:-3]
        delay_values = obs[-3:]
    else:
        action_values = obs[4:]
        delay_values = None
    actions = action_values.reshape(-1, 2) if len(action_values) else np.zeros((0, 2), dtype=np.float32)
    return base_obs, actions, delay_values


def estimate_state(obs, variant, spec, action_scale):
    if variant == "unseen":
        base_obs = np.asarray(obs, dtype=np.float32)
        return base_obs[:2], base_obs[2:4]

    include_delay_values = variant == "augmented_action_delay"
    base_obs, actions, delay_values = split_augmented_obs(obs, include_delay_values)
    delayed_position = base_obs[:2]
    goal = base_obs[2:4]

    if include_delay_values and delay_values is not None:
        alpha = int(round(float(delay_values[0])))
        beta = int(round(float(delay_values[2])))
        replay_len = max(0, min(len(actions), alpha + beta))
    elif "stochastic" in spec.kind and spec.act_max == 0:
        replay_len = max(0, min(len(actions), int(round(0.25 * spec.obs_max))))
    else:
        replay_len = max(0, min(len(actions), spec.nominal_total_delay))

    # The newest action has just been selected by the remote-brain side of the
    # wrapper and has not yet influenced the returned observation.  Skipping it
    # keeps the simple predictor aligned with the wrapper's one-step action
    # scheduling.
    if replay_len <= 0:
        correction = np.zeros(2, dtype=np.float32)
    else:
        correction = action_scale * np.sum(actions[1 : replay_len + 1], axis=0)
    return delayed_position + correction, goal


def controller_action(obs, variant, spec, args):
    position, goal = estimate_state(obs, variant, spec, args.action_scale)
    action = args.kp * (goal - position)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_episode(spec, variant, seed, episode, args):
    random.seed(seed * 10_000 + episode)
    np.random.seed(seed * 10_000 + episode)
    env = make_delay_env(spec, variant, seed + episode, args)
    obs, _ = env.reset(seed=seed + episode)
    terminated = truncated = False
    distances = []
    rewards = []
    successes = []

    while not (terminated or truncated):
        action = controller_action(obs, variant, spec, args)
        obs, reward, terminated, truncated, info = env.step(action)
        base_env = env.unwrapped
        distance = float(np.linalg.norm(base_env.position - base_env.goal))
        distances.append(distance)
        rewards.append(float(reward))
        successes.append(float(info.get("is_success", distance <= args.success_threshold)))

    env.close()
    return {
        "episode_steps": len(rewards),
        "episode_return": float(np.sum(rewards)),
        "mean_distance": float(np.mean(distances)),
        "final_distance": float(distances[-1]),
        "min_distance": float(np.min(distances)),
        "success": float(successes[-1]),
        "ever_success": float(np.max(successes)),
    }


def build_specs(args):
    specs = []
    for delay in args.constant_delays:
        specs.append(("constant_sweep", DelaySpec("constant", delay, delay, delay, delay), float(delay)))
    for upper in args.stochastic_uppers:
        specs.append(("stochastic_sweep", DelaySpec("stochastic", 0, 0, 0, upper), float(upper)))
    specs.append(("state_constant", DelaySpec("state_constant", args.state_constant_delay, args.state_constant_delay, args.state_constant_delay, args.state_constant_delay), float(args.state_constant_delay)))
    specs.append(("state_stochastic", DelaySpec("state_stochastic", 0, 0, 0, args.state_stochastic_upper), float(args.state_stochastic_upper)))
    return specs


def run_demo(args):
    rows = []
    for experiment, spec, x_value in build_specs(args):
        for variant in VARIANT_ORDER:
            for seed in args.seeds:
                for episode in range(args.episodes):
                    metrics = run_episode(spec, variant, seed, episode, args)
                    rows.append(
                        {
                            "experiment": experiment,
                            "variant": variant,
                            "variant_label": VARIANTS[variant],
                            "delay_label": spec.label,
                            "x_value": x_value,
                            "act_min": spec.act_min,
                            "act_max": spec.act_max,
                            "obs_min": spec.obs_min,
                            "obs_max": spec.obs_max,
                            "seed": seed,
                            "episode": episode,
                            **metrics,
                        }
                    )
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


def plot_sweep(ax, rows, experiment, metric, title, xlabel):
    subset = [row for row in rows if row["experiment"] == experiment]
    summary = aggregate(subset, ["variant", "x_value"], metric)
    for variant in VARIANT_ORDER:
        part = sorted([row for row in summary if row["variant"] == variant], key=lambda item: item["x_value"])
        if not part:
            continue
        xs = np.asarray([row["x_value"] for row in part], dtype=float)
        ys = np.asarray([row["iqm"] for row in part], dtype=float)
        q1 = np.asarray([row["q1"] for row in part], dtype=float)
        q3 = np.asarray([row["q3"] for row in part], dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=COLORS[variant], label=VARIANTS[variant])
        ax.fill_between(xs, q1, q3, color=COLORS[variant], alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label(metric))


def plot_state_bars(ax, rows, experiment, metric, title):
    subset = [row for row in rows if row["experiment"] == experiment]
    summary = aggregate(subset, ["variant"], metric)
    values = {row["variant"]: row for row in summary}
    xs = np.arange(len(VARIANT_ORDER))
    heights = []
    lower = []
    upper = []
    colors = []
    for variant in VARIANT_ORDER:
        row = values[variant]
        heights.append(row["iqm"])
        lower.append(max(0.0, row["iqm"] - row["q1"]))
        upper.append(max(0.0, row["q3"] - row["iqm"]))
        colors.append(COLORS[variant])
    ax.bar(xs, heights, color=colors, width=0.72)
    ax.errorbar(xs, heights, yerr=[lower, upper], fmt="none", color="black", capsize=4, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Unseen", "+ actions", "+ actions\n+ delays"])
    ax.set_title(title)
    ax.set_ylabel(metric_label(metric))


def write_summary(path, rows, metric):
    group_cols = ["experiment", "variant", "variant_label", "delay_label", "x_value"]
    summary = aggregate(rows, group_cols, metric)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = group_cols + ["iqm", "q1", "q3", "n"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def plot_results(output_dir, rows, metric):
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    plot_sweep(
        axes[0],
        rows,
        "constant_sweep",
        metric,
        "Constant Action + Observation Delay",
        "Delay pair (a,o), a=o",
    )
    plot_sweep(
        axes[1],
        rows,
        "stochastic_sweep",
        metric,
        "Stochastic Observation Delay",
        "Observation-delay upper bound",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(output_dir / "delay_length_impact.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    plot_state_bars(axes[0], rows, "state_constant", metric, "State Information: Constant Delay")
    plot_state_bars(axes[1], rows, "state_stochastic", metric, "State Information: Stochastic Delay")
    fig.savefig(output_dir / "state_information.png", bbox_inches="tight")
    plt.close(fig)


def write_process_doc(path, args):
    payload = {
        "environment": "PointReachEnv",
        "delay_wrapper": "wrappers_rd.RandomDelayWrapper via UnseenRandomDelayWrapper/AugmentedRandomDelayWrapper/AugmentedDelayInfoWrapper",
        "state": "[x, y, goal_x, goal_y]",
        "action": "2D clipped velocity",
        "metric": metric_label(args.metric),
        "constant_delays": args.constant_delays,
        "stochastic_observation_uppers": args.stochastic_uppers,
        "state_constant_delay": args.state_constant_delay,
        "state_stochastic_upper": args.state_stochastic_upper,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "horizon": args.horizon,
        "assumption": "This is a controlled delay-mechanism demonstration, not a replacement for Fetch policy training.",
    }
    text = [
        "# Simple Delay-Type Demonstration",
        "",
        "This run uses a minimal 2D point-reaching environment to isolate the delay mechanism.",
        "The same random-delay wrappers used by the Fetch experiments are applied around the environment.",
        "",
        "## Setup",
        "",
        "- Environment state: `[x, y, goal_x, goal_y]`.",
        "- Action: clipped 2D velocity.",
        "- Reward: negative Euclidean distance to the goal.",
        f"- Evaluation metric: {metric_label(args.metric)} from the base environment state.",
        "- Controller: proportional controller over the estimated current state.",
        "- Unseen delay observes only the delayed state.",
        "- Augmented (+action buffer) reconstructs state from delayed state plus recent actions using the nominal delay.",
        "- Augmented (+action buffer + delay values) reconstructs using the sampled delay values exposed by the wrapper.",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "## Outputs",
        "",
        "- `delay_length_impact.png`: constant and stochastic delay sweeps.",
        "- `state_information.png`: unseen vs augmented state information at fixed delay settings.",
        "- `results.csv`: per-episode results.",
        "- `summary.csv`: IQM/IQR summary used for the plots.",
    ]
    path.write_text("\n".join(text) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="diagnostics/simple_delay_type_demo")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--constant-delays", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--stochastic-uppers", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--state-constant-delay", type=int, default=15)
    parser.add_argument("--state-stochastic-upper", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--process-noise", type=float, default=0.005)
    parser.add_argument("--goal-noise", type=float, default=0.08)
    parser.add_argument("--success-threshold", type=float, default=0.06)
    parser.add_argument("--kp", type=float, default=3.5)
    parser.add_argument("--metric", choices=["final_distance", "mean_distance", "min_distance"], default="mean_distance")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = run_demo(args)
    write_csv(output_dir / "results.csv", rows)
    write_summary(output_dir / "summary.csv", rows, args.metric)
    plot_results(output_dir, rows, args.metric)
    write_process_doc(output_dir / "PROCESS.md", args)
    print(f"Saved results: {output_dir / 'results.csv'}")
    print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(f"Saved plots: {output_dir / 'delay_length_impact.png'}, {output_dir / 'state_information.png'}")
    print(f"Saved process doc: {output_dir / 'PROCESS.md'}")


if __name__ == "__main__":
    main()
