"""Small FetchReach convergence diagnostic for delayed-state variants."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from fetch_delay_experiments import (
    build_delay_env,
    constant_delay,
    make_base_env,
    seed_everything,
)


VARIANT_LABELS = {
    "no_delay": "No delay",
    "unseen": "Unseen delay",
    "augmented_action": "Augmented (+action buffer)",
}


def build_env(env_id, variant, delay, seed):
    if variant == "no_delay":
        return make_base_env(env_id, seed)
    setting = constant_delay(delay, delay)
    return build_delay_env(
        env_id,
        variant,
        setting,
        seed,
        act_buffer_len=setting.act_buffer_len,
        obs_buffer_len=setting.obs_buffer_len,
    )


def evaluate(model, env_id, variant, delay, seed, episodes):
    env = build_env(env_id, variant, delay, seed)
    returns = []
    lengths = []
    successes = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        rewards = []
        last_info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            rewards.append(float(reward))

        returns.append(float(np.sum(rewards)))
        lengths.append(len(rewards))
        successes.append(float(last_info.get("is_success", 0.0)))

    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "episode_len_mean": float(np.mean(lengths)),
    }


def train_variant(args, variant):
    seed_everything(args.seed)
    env = build_env(args.env_id, variant, args.delay, args.seed)
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        buffer_size=args.buffer_size,
        learning_starts=min(args.learning_starts, max(1, args.steps - 1)),
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    rows = []
    for step in range(args.eval_every, args.steps + 1, args.eval_every):
        model.learn(
            total_timesteps=args.eval_every,
            reset_num_timesteps=(step == args.eval_every),
            log_interval=args.log_interval,
        )
        metrics = evaluate(
            model,
            args.env_id,
            variant,
            args.delay,
            args.eval_seed + step,
            args.n_eval_episodes,
        )
        row = {
            "env_id": args.env_id,
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "seed": args.seed,
            "delay": args.delay,
            "steps": step,
            **metrics,
        }
        rows.append(row)
        print(
            f"{variant:16s} step={step:7d} "
            f"success={metrics['success_rate']:.3f} "
            f"return={metrics['return_mean']:.3f} "
            f"len={metrics['episode_len_mean']:.1f}",
            flush=True,
        )

    env.close()
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env_id",
        "variant",
        "variant_label",
        "seed",
        "delay",
        "steps",
        "success_rate",
        "return_mean",
        "return_std",
        "episode_len_mean",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    for variant in dict.fromkeys(row["variant"] for row in rows):
        part = [row for row in rows if row["variant"] == variant]
        steps = [row["steps"] for row in part]
        axes[0].plot(steps, [row["success_rate"] for row in part], marker="o", label=VARIANT_LABELS[variant])
        axes[1].plot(steps, [row["return_mean"] for row in part], marker="o", label=VARIANT_LABELS[variant])

    axes[0].set_title("Evaluation Success")
    axes[0].set_xlabel("Training steps")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Evaluation Return")
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("Mean return")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="FetchReach-v2")
    parser.add_argument("--variants", nargs="+", choices=VARIANT_LABELS, default=["unseen", "augmented_action"])
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="diagnostics/fetchreach_convergence")
    args = parser.parse_args()

    torch.set_num_threads(1)
    rows = []
    for variant in args.variants:
        rows.extend(train_variant(args, variant))

    output_dir = Path(args.output_dir)
    stem = f"{args.env_id}_seed{args.seed}_constant{args.delay}"
    csv_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"
    write_csv(csv_path, rows)
    plot_rows(plot_path, rows)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
