"""
Run the FetchPush delay experiment pipeline end to end.

This orchestrates optional preflight smoke testing, training/evaluation,
plotting, and strict coverage checks with one set of shared arguments.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from fetch_delay_experiments import EXPERIMENTS, VARIANTS


ROOT = Path(__file__).resolve().parent


def default_prefix(env_id):
    if env_id.startswith("FetchReach"):
        return "FetchReach"
    if env_id.startswith("FetchPush"):
        return "FetchPush"
    if env_id.startswith("FetchSlide"):
        return "FetchSlide"
    if env_id.startswith("FetchPickAndPlace"):
        return "FetchPickAndPlace"
    return env_id.replace("-RemotePDNorm-v0", "").replace("-v0", "").replace("-v2", "")


def run(cmd):
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_if_value(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def add_list(cmd, flag, values):
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def has_glob(value):
    return value is not None and any(char in str(value) for char in "*?[")


def add_wandb_args(cmd, args):
    if not args.wandb:
        return
    cmd.extend(["--wandb", "--wandb-project", args.wandb_project, "--wandb-mode", args.wandb_mode])
    add_if_value(cmd, "--wandb-entity", args.wandb_entity)
    add_if_value(cmd, "--wandb-group", args.wandb_group)
    add_if_value(cmd, "--wandb-run-name-prefix", args.wandb_run_name_prefix)


def require_wandb_available():
    result = subprocess.run(
        [sys.executable, "-c", "import wandb"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(
            "error: --wandb was requested, but wandb is not installed in this interpreter.\n"
            "Install/login before launching the long run:\n"
            "  uv --cache-dir /tmp/uv-cache pip install --python .venv/bin/python wandb\n"
            "  .venv/bin/wandb login\n"
            f"\nPython stderr:\n{result.stderr.strip()}"
        )


def matrix_args(args):
    cmd = []
    add_list(cmd, "--experiments", args.experiments)
    add_list(cmd, "--variants", args.variants)
    add_list(cmd, "--constant-delays", args.constant_delays)
    add_list(cmd, "--stochastic-obs-uppers", args.stochastic_obs_uppers)
    add_if_value(cmd, "--state-constant-delay", args.state_constant_delay)
    add_if_value(cmd, "--state-stochastic-upper", args.state_stochastic_upper)
    add_list(cmd, "--action-obs-delays", args.action_obs_delays)
    add_if_value(cmd, "--generalization-constant-train", args.generalization_constant_train)
    add_list(cmd, "--generalization-constant-eval", args.generalization_constant_eval)
    add_if_value(cmd, "--generalization-stochastic-train-upper", args.generalization_stochastic_train_upper)
    add_list(cmd, "--generalization-stochastic-eval-uppers", args.generalization_stochastic_eval_uppers)
    return cmd


def fetch_args(args, output_dir, results_csv, shard_count=None, shard_index=None):
    shard_count = args.shard_count if shard_count is None else shard_count
    shard_index = args.shard_index if shard_index is None else shard_index
    cmd = [
        sys.executable,
        "fetch_delay_experiments.py",
        "--env-id",
        args.env_id,
        "--output-dir",
        str(output_dir),
        "--results-csv",
        str(results_csv),
        "--steps",
        str(args.steps),
        "--n-eval-episodes",
        str(args.n_eval_episodes),
        "--device",
        args.device,
        "--buffer-size",
        str(args.buffer_size),
        "--batch-size",
        str(args.batch_size),
        "--learning-starts",
        str(args.learning_starts),
        "--log-interval",
        str(args.log_interval),
    ]
    add_list(cmd, "--seeds", args.seeds)
    add_if_value(cmd, "--operator-model", args.operator_model)
    if args.manifest and not has_glob(args.manifest):
        add_if_value(cmd, "--manifest", args.manifest)
    if shard_count > 1:
        cmd.extend(["--shard-count", str(shard_count), "--shard-index", str(shard_index)])
    cmd.extend(matrix_args(args))
    if args.tensorboard:
        cmd.append("--tensorboard")
    if args.force_train:
        cmd.append("--force-train")
    if args.force_eval:
        cmd.append("--force-eval")
    if args.eval_only:
        cmd.append("--eval-only")
    add_wandb_args(cmd, args)
    if args.wandb and not args.wandb_model_artifacts:
        cmd.append("--no-wandb-model-artifacts")
    return cmd


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    parser.add_argument("--operator-model", default=None)
    parser.add_argument("--output-dir", default="fetch_delay_runs")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--manifest", default=None, help="Manifest path or glob used for training metadata and checking.")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--allow-partial-manifest", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--run-shards-locally",
        type=int,
        default=0,
        help="Run N deterministic shards sequentially, then plot/check all shard manifests.",
    )
    parser.add_argument("--metric", default="episode_return", choices=["episode_return", "episode_mean_reward", "episode_mean_distance"])

    parser.add_argument("--wandb", action="store_true", help="Log evaluation jobs and aggregate plots directly to W&B.")
    parser.add_argument("--wandb-project", default="fetch-delay")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name-prefix", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-model-artifacts", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS.keys(), default=list(VARIANTS.keys()))
    parser.add_argument("--constant-delays", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--stochastic-obs-uppers", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--state-constant-delay", type=int, default=20)
    parser.add_argument("--state-stochastic-upper", type=int, default=20)
    parser.add_argument("--action-obs-delays", nargs="+", type=int, default=[0, 10, 20, 30, 40])
    parser.add_argument("--generalization-constant-train", type=int, default=10)
    parser.add_argument("--generalization-constant-eval", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--generalization-stochastic-train-upper", type=int, default=10)
    parser.add_argument("--generalization-stochastic-eval-uppers", nargs="+", type=int, default=[5, 10, 15, 20])
    return parser


def main():
    args = build_parser().parse_args()
    if args.prefix is None:
        args.prefix = default_prefix(args.env_id)
    output_dir = Path(args.output_dir)
    results_csv = Path(args.results_csv) if args.results_csv else output_dir / "evaluations.csv"
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"

    if args.run_shards_locally < 0:
        raise SystemExit("error: --run-shards-locally must be >= 0")
    if args.shard_count < 1:
        raise SystemExit("error: --shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("error: --shard-index must satisfy 0 <= index < shard-count")
    if args.run_shards_locally and args.shard_count != 1:
        raise SystemExit("error: use either --run-shards-locally or --shard-count/--shard-index, not both")
    if (
        not args.skip_train
        and args.manifest
        and not has_glob(args.manifest)
        and (args.run_shards_locally or args.shard_count > 1)
    ):
        raise SystemExit("error: omit --manifest when training shards so each shard can write its own manifest")
    if args.wandb:
        require_wandb_available()
    if not args.skip_smoke:
        run([sys.executable, "smoke_fetch_delay_pipeline.py"])

    if args.manifest:
        manifest_arg = args.manifest
    elif args.run_shards_locally:
        manifest_arg = str(output_dir / f"run_manifest_shard*-of-{args.run_shards_locally}.json")
    elif args.shard_count > 1:
        manifest_arg = output_dir / f"run_manifest_shard{args.shard_index}-of-{args.shard_count}.json"
    else:
        manifest_arg = output_dir / "run_manifest.json"

    if not args.skip_train:
        if args.run_shards_locally:
            for shard_index in range(args.run_shards_locally):
                run(fetch_args(args, output_dir, results_csv, shard_count=args.run_shards_locally, shard_index=shard_index))
        else:
            run(fetch_args(args, output_dir, results_csv))

    if not args.skip_plot:
        plot_cmd = [
            sys.executable,
            "plot_fetch_delay_experiments.py",
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(plots_dir),
            "--prefix",
            args.prefix,
            "--metric",
            args.metric,
        ]
        add_wandb_args(plot_cmd, args)
        run(plot_cmd)

    if not args.skip_check:
        check_cmd = [
            sys.executable,
            "check_fetch_delay_results.py",
            "--env-id",
            args.env_id,
            "--results-csv",
            str(results_csv),
            "--manifest",
            str(manifest_arg),
            "--plots-dir",
            str(plots_dir),
            "--prefix",
            args.prefix,
            "--n-eval-episodes",
            str(args.n_eval_episodes),
        ]
        add_list(check_cmd, "--seeds", args.seeds)
        if args.allow_partial_manifest:
            check_cmd.append("--allow-partial-manifest")
        check_cmd.extend(matrix_args(args))
        run(check_cmd)


if __name__ == "__main__":
    main()
