import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch

import robo_local_remote_env  # noqa: F401 - registers the local-remote Gym envs.
from PMDC_wrapper import PMDC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from wrappers_rd import AugmentedRandomDelayWrapper, UnseenRandomDelayWrapper


STEP_MS = 10
DEFAULT_DELAY_RANGES = ("90-120", "250-290")
DEFAULT_ALGORITHMS = ("PMDC", "A-SAC", "SAC")


@dataclass(frozen=True)
class DelayConfig:
    label: str
    min_ms: int
    max_ms: int
    act_delay: int
    obs_delay_range: range
    act_delay_range: range


def parse_delay_range(value):
    try:
        start, stop = [int(part) for part in value.lower().replace("ms", "").split("-", 1)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a range like 90-120, got {value!r}") from exc
    return delay_config_from_ms(start, stop)


def delay_config_from_ms(min_ms, max_ms):
    if min_ms > max_ms:
        raise argparse.ArgumentTypeError(f"Delay range start must be <= end: {min_ms}-{max_ms}")
    if min_ms % STEP_MS != 0 or max_ms % STEP_MS != 0:
        raise argparse.ArgumentTypeError(f"Delay range must use {STEP_MS} ms increments: {min_ms}-{max_ms}")

    obs_delay_steps = ((max_ms - min_ms) // STEP_MS) + 1
    act_delay = (min_ms // STEP_MS) - 1
    if act_delay < 1:
        raise argparse.ArgumentTypeError(f"Delay range is too small for the existing wrapper: {min_ms}-{max_ms}")

    return DelayConfig(
        label=f"{min_ms}-{max_ms}ms",
        min_ms=min_ms,
        max_ms=max_ms,
        act_delay=act_delay,
        obs_delay_range=range(0, obs_delay_steps),
        act_delay_range=range(act_delay - 1, act_delay),
    )


class TensorboardCallback(BaseCallback):
    """Log delayed reward signals from the nested delay wrappers."""

    def __init__(self, verbose=1, env=None, csv_path=None, record_freq=100):
        super().__init__(verbose)
        self.env = env
        self.csv_path = Path(csv_path) if csv_path else None
        self.record_freq = record_freq
        self.rows = []

    def _on_step(self) -> bool:
        # Use rew_count (episodic) and reward (per-step) from REnvPDNormObs for all algorithms.
        # Searching for "reward" skips PMDC.d_reward and finds REnvPDNormObs.reward,
        # giving a consistent actual tracking metric regardless of algorithm.
        tracking_reward = self._find_attr(("reward",))
        episode_tracking = self._find_attr(("rew_count",))

        if tracking_reward is not None:
            self.logger.record("track/step_reward", tracking_reward)
        if episode_tracking is not None:
            self.logger.record("track/episode_reward", episode_tracking)

        if self.csv_path and self.n_calls % self.record_freq == 0:
            self.rows.append(
                {
                    "step": self.n_calls,
                    "delayed_reward": tracking_reward,
                    "delayed_episode_reward": episode_tracking,
                }
            )

        return True

    def _on_training_end(self) -> None:
        if not self.csv_path:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("step", "delayed_reward", "delayed_episode_reward"))
            writer.writeheader()
            writer.writerows(self.rows)

    def _find_attr(self, names):
        env = self.env
        visited = set()
        while env is not None and id(env) not in visited:
            visited.add(id(env))
            for name in names:
                if hasattr(env, name):
                    return getattr(env, name)
            env = getattr(env, "env", None)
        return None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id, seed=None, operator_model=None):
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed
    if operator_model is not None:
        kwargs["operator_model"] = operator_model

    try:
        env = gym.make(env_id, **kwargs)
    except TypeError:
        env = gym.make(env_id)

    return env


def augmented_delay_v(env, obs_delay_range, act_delay_range):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_history = obs_delay_range.stop + act_delay_range.stop - 1
    augmented_obs_dim = obs_dim + (action_dim * action_history)
    return augmented_obs_dim - 8


def build_training_env(algorithm, env_id, delay_config, seed, n_models, pretrain, operator_model):
    base_env = make_env(env_id, seed=seed, operator_model=operator_model)

    if algorithm == "SAC":
        return UnseenRandomDelayWrapper(
            base_env,
            obs_delay_range=delay_config.obs_delay_range,
            act_delay_range=delay_config.act_delay_range,
        )

    if algorithm == "A-SAC":
        delay_v = augmented_delay_v(base_env, delay_config.obs_delay_range, delay_config.act_delay_range)
        return AugmentedRandomDelayWrapper(
            base_env,
            obs_delay_range=delay_config.obs_delay_range,
            act_delay_range=delay_config.act_delay_range,
            delay_v=delay_v,
        )

    if algorithm == "PMDC":
        fixed_action_env = UnseenRandomDelayWrapper(
            base_env,
            obs_delay_range=range(0, 1),
            act_delay_range=delay_config.act_delay_range,
        )
        corrected_env = PMDC(
            fixed_action_env,
            delay=delay_config.act_delay,
            env_id=env_id,
            pretrain=pretrain,
            n_models=n_models,
        )
        obs_only_delay_range = range(0, delay_config.obs_delay_range.stop)
        delay_v = augmented_delay_v(corrected_env, obs_only_delay_range, range(0, 1))
        return AugmentedRandomDelayWrapper(
            corrected_env,
            obs_delay_range=obs_only_delay_range,
            act_delay_range=range(0, 1),
            delay_v=delay_v,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def train(
    algorithm,
    env_id,
    delay_config,
    steps=80_000,
    seed=0,
    pretrain=False,
    n_models=5,
    device="auto",
    operator_model=None,
    output_dir=".",
    skip_existing=False,
):
    seed_everything(seed)
    output_dir = Path(output_dir)
    settings = f"{n_models}{'_pre' if pretrain else ''}"
    safe_env_id = env_id.replace("/", "_").replace(".", "_")
    model_dir = output_dir / "pred_rl_models" / algorithm / settings / delay_config.label / f"seed_{seed}"
    model_path = model_dir / f"{delay_config.act_delay}_step_SAC_{safe_env_id}.zip"
    csv_path = output_dir / "pred_results" / algorithm / settings / delay_config.label / f"seed_{seed}.csv"

    if skip_existing and model_path.exists():
        print(f"[skip] {algorithm} {delay_config.label} seed={seed}: {model_path}")
        return model_path

    print(
        f"[run] algorithm={algorithm} range={delay_config.label} "
        f"act_delay={delay_config.act_delay} obs_delay={delay_config.obs_delay_range} seed={seed}"
    )
    env = build_training_env(algorithm, env_id, delay_config, seed, n_models, pretrain, operator_model)

    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = output_dir / "pred_logs" / algorithm / settings / delay_config.label / f"seed_{seed}"
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=str(tensorboard_log),
        buffer_size=20_000,
        device=device,
        learning_starts=20_000,
    )
    model.learn(total_timesteps=steps, log_interval=1, callback=TensorboardCallback(env=env, csv_path=csv_path))
    model.save(str(model_path))
    env.close()
    return model_path


def build_parser():
    parser = argparse.ArgumentParser(description="Recreate PMDC vs A-SAC vs SAC delay-range results.")
    parser.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    parser.add_argument(
        "--delay-ranges",
        nargs="+",
        type=parse_delay_range,
        default=[parse_delay_range(delay_range) for delay_range in DEFAULT_DELAY_RANGES],
    )
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_ALGORITHMS, default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--operator-model", default=None, help="Path to the trained operator SAC checkpoint.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for delay_config in args.delay_ranges:
        for algorithm in args.algorithms:
            train(
                algorithm=algorithm,
                env_id=args.env_id,
                delay_config=delay_config,
                steps=args.steps,
                seed=args.seed,
                pretrain=args.pretrain,
                n_models=args.n_models,
                device=args.device,
                operator_model=args.operator_model,
                output_dir=args.output_dir,
                skip_existing=args.skip_existing,
            )


if __name__ == "__main__":
    main()
