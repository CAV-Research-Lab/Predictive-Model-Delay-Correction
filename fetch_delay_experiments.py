"""
Run the FetchPush delayed-observation/action experiments.

The delay pair notation used here is (action delay, observation delay), in
environment steps. Constant delays use a single exact value; stochastic
observation-delay settings sample uniformly from 0..upper.
"""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch

import robo_local_remote_env  # noqa: F401 - registers FetchPush-RemotePDNorm-v0.
from stable_baselines3 import SAC
from wrappers_rd import (
    AugmentedDelayInfoWrapper,
    AugmentedRandomDelayWrapper,
    UnseenRandomDelayWrapper,
)


VARIANTS = {
    "unseen": "Unseen delay",
    "augmented_action": "Augmented (+action buffer)",
    "augmented_action_delay": "Augmented (+action buffer + delay values)",
}

RESULT_FIELDNAMES = (
    "env_id",
    "experiment",
    "comparison",
    "series",
    "variant",
    "variant_label",
    "train_delay",
    "train_delay_name",
    "eval_delay",
    "eval_delay_name",
    "x_value",
    "x_label",
    "act_buffer_len",
    "obs_buffer_len",
    "seed",
    "episode",
    "episode_steps",
    "episode_return",
    "episode_mean_reward",
    "episode_mean_distance",
    "model_path",
)

NUMERIC_COLUMNS = (
    "episode_steps",
    "episode_return",
    "episode_mean_reward",
    "episode_mean_distance",
)

EXPERIMENTS = (
    "constant_sweep",
    "stochastic_sweep",
    "state_constant",
    "state_stochastic",
    "action_vs_observation",
    "generalization_constant",
    "generalization_stochastic",
)


@dataclass(frozen=True)
class DelaySetting:
    name: str
    display: str
    act_min: int
    act_max: int
    obs_min: int
    obs_max: int
    stochastic: bool = False

    @property
    def act_range(self):
        return range(self.act_min, self.act_max + 1)

    @property
    def obs_range(self):
        return range(self.obs_min, self.obs_max + 1)

    @property
    def act_buffer_len(self):
        return self.act_max + 1

    @property
    def obs_buffer_len(self):
        return self.obs_max + 1


@dataclass(frozen=True)
class EvalSpec:
    experiment: str
    comparison: str
    setting: DelaySetting
    x_value: float
    x_label: str
    series: str


@dataclass(frozen=True)
class RunSpec:
    variant: str
    train_setting: DelaySetting
    eval_specs: tuple[EvalSpec, ...]
    act_buffer_len: int
    obs_buffer_len: int

    @property
    def train_id(self):
        return sanitize(
            f"{self.variant}__{self.train_setting.name}"
            f"__actbuf{self.act_buffer_len}__obsbuf{self.obs_buffer_len}"
        )


def sanitize(value):
    return (
        value.replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace("/", "_")
    )


def short_env(env_id):
    if env_id.startswith("FetchPush"):
        return "FetchPush"
    return env_id.replace("-RemotePDNorm-v0", "")


def import_wandb():
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in this interpreter. Install it with:\n"
            "  uv --cache-dir /tmp/uv-cache pip install --python .venv/bin/python wandb"
        ) from exc
    return wandb


def safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def finite(values):
    return [value for value in values if math.isfinite(value)]


def mean(values):
    values = finite(values)
    return sum(values) / len(values) if values else math.nan


def quantile(values, q):
    values = sorted(finite(values))
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def iqm(values):
    values = finite(values)
    if not values:
        return math.nan
    if len(values) < 4:
        return mean(values)
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    middle = [value for value in values if q1 <= value <= q3]
    return mean(middle or values)


def default_wandb_group(args):
    seeds = "-".join(str(seed) for seed in args.seeds)
    return f"{short_env(args.env_id)}-{args.steps}steps-{args.n_eval_episodes}evaleps-seeds{seeds}"


def wandb_eval_run_id(args, spec, eval_spec, seed, model_path):
    raw = "|".join(
        str(part)
        for part in (
            args.env_id,
            args.steps,
            args.n_eval_episodes,
            seed,
            spec.variant,
            eval_spec.experiment,
            eval_spec.comparison,
            spec.train_setting.name,
            eval_spec.setting.name,
            model_path,
        )
    )
    return f"fetch-delay-eval-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def wandb_model_run_id(args, spec, seed, model_path):
    raw = "|".join(
        str(part)
        for part in (
            args.env_id,
            args.steps,
            seed,
            spec.variant,
            spec.train_setting.name,
            spec.act_buffer_len,
            spec.obs_buffer_len,
            model_path,
        )
    )
    return f"fetch-delay-model-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def wandb_eval_run_name(args, spec, eval_spec, seed):
    name = (
        f"{short_env(args.env_id)} seed{seed} {spec.variant} "
        f"{eval_spec.experiment}/{eval_spec.comparison} "
        f"train={spec.train_setting.display} eval={eval_spec.setting.display}"
    )
    return f"{args.wandb_run_name_prefix} | {name}" if args.wandb_run_name_prefix else name


def wandb_model_run_name(args, spec, seed):
    name = (
        f"{short_env(args.env_id)} seed{seed} {spec.variant} "
        f"checkpoint train={spec.train_setting.display}"
    )
    return f"{args.wandb_run_name_prefix} | {name}" if args.wandb_run_name_prefix else name


def wandb_marker_dir(output_dir, env_id, seed, spec):
    return model_dir_for(output_dir, env_id, seed, spec) / "wandb"


def wandb_eval_marker_path(args, spec, eval_spec, seed, model_path):
    run_id = wandb_eval_run_id(args, spec, eval_spec, seed, model_path)
    return wandb_marker_dir(args.output_dir, args.env_id, seed, spec) / f"{run_id}.json"


def wandb_model_marker_path(args, spec, seed, model_path):
    run_id = wandb_model_run_id(args, spec, seed, model_path)
    return wandb_marker_dir(args.output_dir, args.env_id, seed, spec) / f"{run_id}.json"


def write_wandb_marker(path, payload, args):
    if args.wandb_mode == "disabled":
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "mode": args.wandb_mode,
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize_eval_rows(rows):
    values = {column: [safe_float(row.get(column)) for row in rows] for column in NUMERIC_COLUMNS}
    distances = values["episode_mean_distance"]
    returns = values["episode_return"]
    return {
        "eval/episode_count": len(rows),
        "eval/episode_steps_mean": mean(values["episode_steps"]),
        "eval/episode_return_mean": mean(returns),
        "eval/episode_return_iqm": iqm(returns),
        "eval/episode_return_q1": quantile(returns, 0.25),
        "eval/episode_return_q3": quantile(returns, 0.75),
        "eval/episode_mean_reward_mean": mean(values["episode_mean_reward"]),
        "eval/episode_mean_distance_mean": mean(distances),
        "eval/episode_mean_distance_iqm": iqm(distances),
        "eval/episode_mean_distance_q1": quantile(distances, 0.25),
        "eval/episode_mean_distance_q3": quantile(distances, 0.75),
        "eval/success_rate_5cm": mean([1.0 if value <= 0.05 else 0.0 for value in finite(distances)]),
        "eval/success_rate_2cm": mean([1.0 if value <= 0.02 else 0.0 for value in finite(distances)]),
    }


def log_eval_rows_to_wandb(args, spec, eval_spec, seed, model_path, rows):
    if not args.wandb or not rows:
        return

    marker_path = wandb_eval_marker_path(args, spec, eval_spec, seed, model_path)
    if marker_path.exists():
        print(f"[wandb eval skip] {marker_path}")
        return

    wandb = import_wandb()
    run_name = wandb_eval_run_name(args, spec, eval_spec, seed)
    run_id = wandb_eval_run_id(args, spec, eval_spec, seed, model_path)
    config = {
        "env_id": args.env_id,
        "seed": seed,
        "steps": args.steps,
        "n_eval_episodes": args.n_eval_episodes,
        "variant": spec.variant,
        "variant_label": VARIANTS[spec.variant],
        "experiment": eval_spec.experiment,
        "comparison": eval_spec.comparison,
        "series": eval_spec.series,
        "train_delay": spec.train_setting.display,
        "train_delay_name": spec.train_setting.name,
        "eval_delay": eval_spec.setting.display,
        "eval_delay_name": eval_spec.setting.name,
        "act_buffer_len": spec.act_buffer_len,
        "obs_buffer_len": spec.obs_buffer_len,
        "x_value": eval_spec.x_value,
        "x_label": eval_spec.x_label,
        "model_path": str(model_path),
        "results_csv": str(args.results_csv),
        "success_thresholds_m": {"success_rate_5cm": 0.05, "success_rate_2cm": 0.02},
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or default_wandb_group(args),
        job_type="evaluation",
        id=run_id,
        name=run_name,
        config=config,
        resume="allow",
        mode=args.wandb_mode,
        reinit="finish_previous",
    )
    run.define_metric("episode")
    run.define_metric("eval/*", step_metric="episode")

    table = wandb.Table(columns=["episode", *NUMERIC_COLUMNS])
    for row in sorted(rows, key=lambda item: int(item["episode"])):
        episode = int(row["episode"])
        metrics = {f"eval/{column}": safe_float(row.get(column)) for column in NUMERIC_COLUMNS}
        run.log({"episode": episode, **metrics}, step=episode)
        table.add_data(episode, *(metrics[f"eval/{column}"] for column in NUMERIC_COLUMNS))
    run.log({"eval/episodes": table})

    for key, value in summarize_eval_rows(rows).items():
        run.summary[key] = value
    run.summary["run_name"] = run_name
    run.finish()
    write_wandb_marker(
        marker_path,
        {"kind": "evaluation", "run_id": run_id, "run_name": run_name, "episode_count": len(rows)},
        args,
    )


def log_model_checkpoint_to_wandb(args, spec, seed, model_path, metadata_path):
    if not args.wandb or not args.wandb_model_artifacts:
        return
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.exists():
        return

    marker_path = wandb_model_marker_path(args, spec, seed, model_path)
    if marker_path.exists():
        print(f"[wandb model skip] {marker_path}")
        return

    wandb = import_wandb()
    run_id = wandb_model_run_id(args, spec, seed, model_path)
    run_name = wandb_model_run_name(args, spec, seed)
    artifact_name = sanitize(f"{args.env_id}__{spec.train_id}__seed_{seed}")[:128]
    metadata = {
        "env_id": args.env_id,
        "seed": seed,
        "steps": args.steps,
        "variant": spec.variant,
        "variant_label": VARIANTS[spec.variant],
        "train_delay": spec.train_setting.display,
        "train_delay_name": spec.train_setting.name,
        "act_buffer_len": spec.act_buffer_len,
        "obs_buffer_len": spec.obs_buffer_len,
        "model_path": str(model_path),
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or default_wandb_group(args),
        job_type="model-checkpoint",
        id=run_id,
        name=run_name,
        config=metadata,
        resume="allow",
        mode=args.wandb_mode,
        reinit="finish_previous",
    )
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description="Final SAC policy checkpoint for a Fetch delay training job.",
        metadata=metadata,
    )
    artifact.add_file(str(model_path), name="model.zip")
    if metadata_path.exists():
        artifact.add_file(str(metadata_path), name="metadata.json")
    run.log_artifact(
        artifact,
        aliases=[
            "latest",
            f"seed-{seed}",
            sanitize(spec.variant),
            sanitize(spec.train_setting.name),
        ],
    )
    run.summary["model_path"] = str(model_path)
    run.summary["artifact_name"] = artifact_name
    run.summary["run_name"] = run_name
    run.finish()
    write_wandb_marker(
        marker_path,
        {"kind": "model", "run_id": run_id, "run_name": run_name, "artifact_name": artifact_name},
        args,
    )


def constant_delay(act_delay, obs_delay):
    return DelaySetting(
        name=f"constant_a{act_delay}_o{obs_delay}",
        display=f"({act_delay},{obs_delay})",
        act_min=act_delay,
        act_max=act_delay,
        obs_min=obs_delay,
        obs_max=obs_delay,
        stochastic=False,
    )


def stochastic_obs_delay(obs_upper, act_delay=0):
    return DelaySetting(
        name=f"stochastic_a{act_delay}_o0-{obs_upper}",
        display=f"({act_delay},0-{obs_upper})",
        act_min=act_delay,
        act_max=act_delay,
        obs_min=0,
        obs_max=obs_upper,
        stochastic=True,
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_base_env(env_id, seed, operator_model=None):
    kwargs = {"seed": seed}
    if operator_model:
        kwargs["operator_model"] = operator_model
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=f".*{env_id} is out of date.*", category=DeprecationWarning)
            env = gym.make(env_id, **kwargs)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=f".*{env_id} is out of date.*", category=DeprecationWarning)
            env = gym.make(env_id)
        env.reset(seed=seed)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


def build_delay_env(env_id, variant, setting, seed, act_buffer_len, obs_buffer_len, operator_model=None):
    base_env = make_base_env(env_id, seed, operator_model=operator_model)
    wrapper_kwargs = {
        "obs_delay_range": setting.obs_range,
        "act_delay_range": setting.act_range,
        "obs_buffer_len": obs_buffer_len,
        "act_buffer_len": act_buffer_len,
    }

    if variant == "unseen":
        return UnseenRandomDelayWrapper(base_env, **wrapper_kwargs)
    if variant == "augmented_action":
        return AugmentedRandomDelayWrapper(base_env, **wrapper_kwargs)
    if variant == "augmented_action_delay":
        return AugmentedDelayInfoWrapper(base_env, **wrapper_kwargs)
    raise ValueError(f"Unknown variant: {variant}")


def max_buffer_lens(settings):
    return max(s.act_buffer_len for s in settings), max(s.obs_buffer_len for s in settings)


def add_train_eval(specs, variant, train_setting, eval_spec, act_buffer_len=None, obs_buffer_len=None):
    act_buffer_len = train_setting.act_buffer_len if act_buffer_len is None else act_buffer_len
    obs_buffer_len = train_setting.obs_buffer_len if obs_buffer_len is None else obs_buffer_len
    specs.append(
        RunSpec(
            variant=variant,
            train_setting=train_setting,
            eval_specs=(eval_spec,),
            act_buffer_len=act_buffer_len,
            obs_buffer_len=obs_buffer_len,
        )
    )


def build_run_specs(args):
    specs = []
    variants = args.variants
    selected = set(args.experiments)

    constant_settings = [constant_delay(delay, delay) for delay in args.constant_delays]
    stochastic_settings = [stochastic_obs_delay(delay) for delay in args.stochastic_obs_uppers]
    action_only_settings = [constant_delay(delay, 0) for delay in args.action_obs_delays]
    obs_only_settings = [constant_delay(0, delay) for delay in args.action_obs_delays]

    for variant in variants:
        if "constant_sweep" in selected:
            for setting in constant_settings:
                add_train_eval(
                    specs,
                    variant,
                    setting,
                    EvalSpec(
                        "constant_sweep",
                        "constant_delay_length",
                        setting,
                        setting.act_max,
                        setting.display,
                        VARIANTS[variant],
                    ),
                )

        if "stochastic_sweep" in selected:
            for setting in stochastic_settings:
                add_train_eval(
                    specs,
                    variant,
                    setting,
                    EvalSpec(
                        "stochastic_sweep",
                        "stochastic_observation_upper",
                        setting,
                        setting.obs_max,
                        setting.display,
                        VARIANTS[variant],
                    ),
                )

        if "state_constant" in selected:
            setting = constant_delay(args.state_constant_delay, args.state_constant_delay)
            add_train_eval(
                specs,
                variant,
                setting,
                EvalSpec("state_constant", "state_info_constant", setting, 0, VARIANTS[variant], VARIANTS[variant]),
            )

        if "state_stochastic" in selected:
            setting = stochastic_obs_delay(args.state_stochastic_upper)
            add_train_eval(
                specs,
                variant,
                setting,
                EvalSpec("state_stochastic", "state_info_stochastic", setting, 0, VARIANTS[variant], VARIANTS[variant]),
            )

        if "action_vs_observation" in selected:
            for setting in action_only_settings:
                add_train_eval(
                    specs,
                    variant,
                    setting,
                    EvalSpec("action_vs_observation", "action_delay_only", setting, setting.act_max, "Action", VARIANTS[variant]),
                )
            for setting in obs_only_settings:
                add_train_eval(
                    specs,
                    variant,
                    setting,
                    EvalSpec(
                        "action_vs_observation",
                        "observation_delay_only",
                        setting,
                        setting.obs_max,
                        "Observation",
                        VARIANTS[variant],
                    ),
                )

        if "generalization_constant" in selected:
            train_setting = constant_delay(args.generalization_constant_train, args.generalization_constant_train)
            eval_settings = [constant_delay(delay, delay) for delay in args.generalization_constant_eval]
            act_buffer_len, obs_buffer_len = max_buffer_lens((train_setting, *eval_settings))
            specs.append(
                RunSpec(
                    variant=variant,
                    train_setting=train_setting,
                    eval_specs=tuple(
                        EvalSpec(
                            "generalization_constant",
                            "train_constant_eval_constant",
                            setting,
                            setting.act_max,
                            setting.display,
                            VARIANTS[variant],
                        )
                        for setting in eval_settings
                    ),
                    act_buffer_len=act_buffer_len,
                    obs_buffer_len=obs_buffer_len,
                )
            )

        if "generalization_stochastic" in selected:
            train_setting = stochastic_obs_delay(args.generalization_stochastic_train_upper)
            eval_settings = [stochastic_obs_delay(delay) for delay in args.generalization_stochastic_eval_uppers]
            act_buffer_len, obs_buffer_len = max_buffer_lens((train_setting, *eval_settings))
            specs.append(
                RunSpec(
                    variant=variant,
                    train_setting=train_setting,
                    eval_specs=tuple(
                        EvalSpec(
                            "generalization_stochastic",
                            "train_stochastic_eval_stochastic",
                            setting,
                            setting.obs_max,
                            setting.display,
                            VARIANTS[variant],
                        )
                        for setting in eval_settings
                    ),
                    act_buffer_len=act_buffer_len,
                    obs_buffer_len=obs_buffer_len,
                )
            )

    return merge_run_specs(specs)


def merge_run_specs(specs):
    merged = {}
    order = []
    for spec in specs:
        key = (spec.variant, spec.train_setting, spec.act_buffer_len, spec.obs_buffer_len)
        if key not in merged:
            merged[key] = spec
            order.append(key)
            continue
        merged[key] = RunSpec(
            variant=spec.variant,
            train_setting=spec.train_setting,
            eval_specs=merged[key].eval_specs + spec.eval_specs,
            act_buffer_len=spec.act_buffer_len,
            obs_buffer_len=spec.obs_buffer_len,
        )
    return [merged[key] for key in order]


def model_dir_for(output_dir, env_id, seed, spec):
    return Path(output_dir) / "models" / sanitize(env_id) / spec.train_id / f"seed_{seed}"


def train_model(args, spec, seed):
    seed_everything(seed)
    model_dir = model_dir_for(args.output_dir, args.env_id, seed, spec)
    model_path = model_dir / "model.zip"
    metadata_path = model_dir / "metadata.json"
    if model_path.exists() and not args.force_train:
        print(f"[train skip] {spec.train_id} seed={seed}")
        log_model_checkpoint_to_wandb(args, spec, seed, model_path, metadata_path)
        return model_path

    print(
        f"[train] variant={spec.variant} train={spec.train_setting.display} "
        f"act_buffer={spec.act_buffer_len} obs_buffer={spec.obs_buffer_len} seed={seed}"
    )
    env = build_delay_env(
        args.env_id,
        spec.variant,
        spec.train_setting,
        seed,
        spec.act_buffer_len,
        spec.obs_buffer_len,
        operator_model=args.operator_model,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = None
    if args.tensorboard:
        tensorboard_log = str(model_dir / "tb")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=tensorboard_log,
        buffer_size=args.buffer_size,
        learning_starts=min(args.learning_starts, max(1, args.steps - 1)),
        batch_size=args.batch_size,
        device=args.device,
        seed=seed,
    )
    model.learn(total_timesteps=args.steps, log_interval=args.log_interval)
    model.save(str(model_path))
    env.close()
    metadata_path.write_text(
        json.dumps(
            {
                "env_id": args.env_id,
                "variant": spec.variant,
                "variant_label": VARIANTS[spec.variant],
                "train_delay": spec.train_setting.__dict__,
                "act_buffer_len": spec.act_buffer_len,
                "obs_buffer_len": spec.obs_buffer_len,
                "seed": seed,
                "steps": args.steps,
            },
            indent=2,
        )
    )
    log_model_checkpoint_to_wandb(args, spec, seed, model_path, metadata_path)
    return model_path


def eval_key_filter(rows, spec, eval_spec, seed):
    return [row for row in rows if eval_key_matches(row, spec, eval_spec, seed)]


def eval_key_matches(row, spec, eval_spec, seed):
    return (
        row.get("variant") == spec.variant
        and row.get("train_delay") == spec.train_setting.display
        and row.get("eval_delay") == eval_spec.setting.display
        and row.get("experiment") == eval_spec.experiment
        and row.get("comparison") == eval_spec.comparison
        and row.get("seed") == str(seed)
    )


def completed_episode_numbers(rows):
    episodes = set()
    for row in rows:
        try:
            episodes.add(int(row["episode"]))
        except (KeyError, TypeError, ValueError):
            continue
    return episodes


def load_existing_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            return list(csv.DictReader(handle))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked_csv(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    with locked_csv(path, "a") as handle:
        handle.seek(0, os.SEEK_END)
        needs_header = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(path, rows):
    path = Path(path)
    with locked_csv(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def remove_eval_rows(path, spec, eval_spec, seed):
    path = Path(path)
    if not path.exists():
        return []
    with locked_csv(path, "r+") as handle:
        rows = list(csv.DictReader(handle))
        rows = [row for row in rows if not eval_key_matches(row, spec, eval_spec, seed)]
        handle.seek(0)
        handle.truncate()
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        return rows


def evaluate_model(args, spec, eval_spec, seed, model_path):
    if args.force_eval:
        existing = remove_eval_rows(args.results_csv, spec, eval_spec, seed)
    else:
        existing = load_existing_rows(args.results_csv)

    previous = eval_key_filter(existing, spec, eval_spec, seed)
    completed_episodes = completed_episode_numbers(previous)
    missing_episodes = [episode for episode in range(args.n_eval_episodes) if episode not in completed_episodes]
    if not missing_episodes:
        print(f"[eval skip] {spec.train_id} -> {eval_spec.setting.display} seed={seed}")
        log_eval_rows_to_wandb(args, spec, eval_spec, seed, model_path, previous)
        return

    if not Path(model_path).exists():
        print(f"[eval missing] {model_path}")
        return

    print(
        f"[eval] experiment={eval_spec.experiment} variant={spec.variant} "
        f"train={spec.train_setting.display} eval={eval_spec.setting.display} seed={seed}"
    )
    eval_seed = seed + args.eval_seed_offset
    env = build_delay_env(
        args.env_id,
        spec.variant,
        eval_spec.setting,
        eval_seed,
        spec.act_buffer_len,
        spec.obs_buffer_len,
        operator_model=args.operator_model,
    )
    model = SAC.load(str(model_path), env=env, device=args.device)

    rows = []
    for episode in missing_episodes:
        obs, _ = env.reset(seed=eval_seed + episode)
        terminated = truncated = False
        rewards = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))

        rewards_arr = np.asarray(rewards, dtype=float)
        rows.append(
            {
                "env_id": args.env_id,
                "experiment": eval_spec.experiment,
                "comparison": eval_spec.comparison,
                "series": eval_spec.series,
                "variant": spec.variant,
                "variant_label": VARIANTS[spec.variant],
                "train_delay": spec.train_setting.display,
                "train_delay_name": spec.train_setting.name,
                "eval_delay": eval_spec.setting.display,
                "eval_delay_name": eval_spec.setting.name,
                "x_value": eval_spec.x_value,
                "x_label": eval_spec.x_label,
                "act_buffer_len": spec.act_buffer_len,
                "obs_buffer_len": spec.obs_buffer_len,
                "seed": seed,
                "episode": episode,
                "episode_steps": len(rewards),
                "episode_return": float(np.sum(rewards_arr)),
                "episode_mean_reward": float(np.mean(rewards_arr)),
                "episode_mean_distance": float(-np.mean(rewards_arr)),
                "model_path": str(model_path),
            }
        )
    env.close()
    append_rows(args.results_csv, rows)
    log_eval_rows_to_wandb(args, spec, eval_spec, seed, model_path, rows)


def print_plan(specs, seeds):
    train_count = len(specs) * len(seeds)
    eval_count = sum(len(spec.eval_specs) for spec in specs) * len(seeds)
    print(f"Planned train jobs: {train_count}")
    print(f"Planned eval jobs:  {eval_count}")
    for spec in specs:
        evals = ", ".join(f"{e.experiment}/{e.comparison}:{e.setting.display}" for e in spec.eval_specs)
        print(
            f"  {spec.variant:22s} train={spec.train_setting.display:10s} "
            f"buffers=({spec.act_buffer_len},{spec.obs_buffer_len}) eval=[{evals}]"
        )


def delay_setting_dict(setting):
    return {
        "name": setting.name,
        "display": setting.display,
        "act_min": setting.act_min,
        "act_max": setting.act_max,
        "obs_min": setting.obs_min,
        "obs_max": setting.obs_max,
        "stochastic": setting.stochastic,
    }


def write_run_manifest(path, specs, seeds, args, all_spec_count=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "env_id": args.env_id,
        "seeds": list(seeds),
        "steps": args.steps,
        "n_eval_episodes": args.n_eval_episodes,
        "device": args.device,
        "experiments": list(args.experiments),
        "variants": list(args.variants),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "spec_count": len(specs),
        "all_spec_count": len(specs) if all_spec_count is None else all_spec_count,
        "train_jobs": len(specs) * len(seeds),
        "eval_jobs": sum(len(spec.eval_specs) for spec in specs) * len(seeds),
        "specs": [
            {
                "variant": spec.variant,
                "variant_label": VARIANTS[spec.variant],
                "train_delay": delay_setting_dict(spec.train_setting),
                "act_buffer_len": spec.act_buffer_len,
                "obs_buffer_len": spec.obs_buffer_len,
                "eval_specs": [
                    {
                        "experiment": eval_spec.experiment,
                        "comparison": eval_spec.comparison,
                        "eval_delay": delay_setting_dict(eval_spec.setting),
                        "x_value": eval_spec.x_value,
                        "x_label": eval_spec.x_label,
                        "series": eval_spec.series,
                    }
                    for eval_spec in spec.eval_specs
                ],
            }
            for spec in specs
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def filter_specs_for_shard(specs, shard_index, shard_count):
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    if shard_count == 1:
        return specs
    return [spec for index, spec in enumerate(specs) if index % shard_count == shard_index]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    parser.add_argument("--operator-model", default=None)
    parser.add_argument("--output-dir", default="fetch_delay_runs")
    parser.add_argument("--results-csv", default="fetch_delay_runs/evaluations.csv")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS.keys(), default=list(VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fetch-delay")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name-prefix", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-model-artifacts", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--constant-delays", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--stochastic-obs-uppers", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--state-constant-delay", type=int, default=20)
    parser.add_argument("--state-stochastic-upper", type=int, default=20)
    parser.add_argument("--action-obs-delays", nargs="+", type=int, default=[0, 10, 20, 30, 40])
    parser.add_argument("--generalization-constant-train", type=int, default=10)
    parser.add_argument("--generalization-constant-eval", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--generalization-stochastic-train-upper", type=int, default=10)
    parser.add_argument("--generalization-stochastic-eval-uppers", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--eval-seed-offset", type=int, default=100_000)
    return parser


def main():
    args = build_parser().parse_args()
    if args.wandb and not args.dry_run:
        import_wandb()
    all_specs = build_run_specs(args)
    try:
        specs = filter_specs_for_shard(all_specs, args.shard_index, args.shard_count)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.shard_count > 1:
        print(f"Shard {args.shard_index}/{args.shard_count}: {len(specs)} of {len(all_specs)} train specs")
    print_plan(specs, args.seeds)
    manifest_path = args.manifest
    if manifest_path is None:
        suffix = "" if args.shard_count == 1 else f"_shard{args.shard_index}-of-{args.shard_count}"
        manifest_path = Path(args.output_dir) / f"run_manifest{suffix}.json"
    manifest_path = Path(manifest_path)
    preserve_existing_manifest = args.eval_only and args.manifest is None and manifest_path.exists()
    if preserve_existing_manifest:
        print(f"Using existing manifest: {manifest_path}")
    else:
        manifest_path = write_run_manifest(manifest_path, specs, args.seeds, args, all_spec_count=len(all_specs))
        print(f"Wrote manifest: {manifest_path}")
    if args.dry_run:
        return

    for seed in args.seeds:
        for spec in specs:
            model_path = model_dir_for(args.output_dir, args.env_id, seed, spec) / "model.zip"
            if not args.eval_only:
                model_path = train_model(args, spec, seed)
            if args.train_only:
                continue
            for eval_spec in spec.eval_specs:
                evaluate_model(args, spec, eval_spec, seed, model_path)


if __name__ == "__main__":
    main()
