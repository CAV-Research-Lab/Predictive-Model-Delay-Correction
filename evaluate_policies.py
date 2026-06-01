"""
Evaluate trained PMDC, A-SAC, and SAC policies on actual delayed environment.
Reports the true tracking distance (not training proxy metrics).

WARNING: This is only valid for SAC and A-SAC. PMDC's ensemble world model is trained
ONLINE inside the wrapper and is NOT saved in the SAC .zip, so reloading a PMDC policy
into a fresh wrapper gives it a RANDOM dynamics model -> the policy receives garbage
future-state predictions and fails catastrophically (~1.9 m). For a valid PMDC number,
use the training-time true-tracking metric (track/step_reward in the TensorBoard logs),
which is also what the thesis reports. See final_summary.py.
"""
import argparse
import numpy as np
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa
import robo_local_remote_env  # noqa

from stable_baselines3 import SAC
from delay_correcting_training import (
    build_training_env, delay_config_from_ms, seed_everything
)


def evaluate_policy(model, env, n_episodes=10, render=False):
    """Run a trained policy for n_episodes and return per-episode tracking distances."""
    episode_distances = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_dist = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_dist.append(abs(reward))  # reward = -distance
            done = terminated or truncated
        episode_distances.append(np.mean(episode_dist))
    return episode_distances


def find_model(output_dir, algorithm, n_models, delay_config, env_id, seed):
    output_dir = Path(output_dir)
    settings = f"{n_models}"
    safe_env_id = env_id.replace("/", "_").replace(".", "_")
    model_path = (
        output_dir / "pred_rl_models" / algorithm / settings
        / delay_config.label / f"seed_{seed}"
        / f"{delay_config.act_delay}_step_SAC_{safe_env_id}"
    )
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    parser.add_argument("--delay", default="250-290", help="Delay range, e.g. 90-120 or 250-290")
    parser.add_argument("--algorithms", nargs="+", default=["PMDC", "A-SAC", "SAC"])
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-model", default=None)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    seed_everything(args.seed)
    delay_min, delay_max = [int(x) for x in args.delay.split("-")]
    delay_config = delay_config_from_ms(delay_min, delay_max)

    print(f"\n=== Policy Evaluation ===")
    print(f"Delay: {delay_config.label}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Act delay steps: {delay_config.act_delay}")
    print()

    results = {}
    for alg in args.algorithms:
        model_path = find_model(args.output_dir, alg, args.n_models, delay_config, args.env_id, args.seed)
        if not (Path(str(model_path) + ".zip")).exists():
            print(f"  [{alg}] Model not found: {model_path}.zip — skip")
            continue

        print(f"  [{alg}] Loading from {model_path}...")
        env = build_training_env(
            alg, args.env_id, delay_config, args.seed,
            args.n_models, False, args.operator_model
        )

        model = SAC.load(str(model_path), env=env)
        distances = evaluate_policy(model, env, n_episodes=args.n_episodes)
        env.close()

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        results[alg] = (mean_dist, std_dist, distances)
        print(f"  [{alg}] Mean distance: {mean_dist:.4f} ± {std_dist:.4f} m")

    print(f"\n=== Summary: {delay_config.label} ===")
    for alg, (mean, std, _) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"  {alg:<8} {mean:.4f} ± {std:.4f} m  (best: {min(_):.4f})")


if __name__ == "__main__":
    main()
