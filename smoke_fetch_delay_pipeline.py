"""
End-to-end smoke test for the FetchPush delay experiment pipeline.

This intentionally trains only tiny 5-step policies under zero delay. It is a
pipeline check, not a meaningful experiment.
"""

import argparse
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

from fetch_delay_experiments import build_delay_env, build_run_specs


ROOT = Path(__file__).resolve().parent


def run(cmd, **kwargs):
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def build_shape_args():
    return Namespace(
        variants=["augmented_action", "augmented_action_delay"],
        experiments=["generalization_constant", "generalization_stochastic"],
        constant_delays=[0, 5, 10, 15, 20],
        stochastic_obs_uppers=[5, 10, 15, 20],
        state_constant_delay=20,
        state_stochastic_upper=20,
        action_obs_delays=[0, 10, 20, 30, 40],
        generalization_constant_train=10,
        generalization_constant_eval=[0, 5, 10, 15, 20],
        generalization_stochastic_train_upper=10,
        generalization_stochastic_eval_uppers=[5, 10, 15, 20],
    )


def check_generalization_shapes():
    specs = build_run_specs(build_shape_args())
    for spec in specs:
        train_env = build_delay_env(
            "FetchPush-RemotePDNorm-v0",
            spec.variant,
            spec.train_setting,
            0,
            spec.act_buffer_len,
            spec.obs_buffer_len,
        )
        train_obs, _ = train_env.reset(seed=0)
        train_shape = train_obs.shape
        train_env.close()

        for eval_spec in spec.eval_specs:
            env = build_delay_env(
                "FetchPush-RemotePDNorm-v0",
                spec.variant,
                eval_spec.setting,
                1,
                spec.act_buffer_len,
                spec.obs_buffer_len,
            )
            obs, _ = env.reset(seed=1)
            env.close()
            if obs.shape != train_shape:
                raise AssertionError(
                    f"Shape mismatch for {spec.variant} train {spec.train_setting.display} "
                    f"eval {eval_spec.setting.display}: {obs.shape} != {train_shape}"
                )
    print("generalization shape check passed")


def check_resume_csv(path):
    df = pd.read_csv(path)
    episodes = sorted(df.loc[df["variant"] == "unseen", "episode"].astype(int).tolist())
    if episodes != [0, 1, 2]:
        raise AssertionError(f"Resume check expected unseen episodes [0, 1, 2], got {episodes}")
    print("resume check passed")


def check_parallel_csv(path):
    text = Path(path).read_text()
    lines = text.splitlines()
    headers = sum(1 for line in lines if line.startswith("env_id,"))
    if headers != 1:
        raise AssertionError(f"Expected one CSV header in {path}, got {headers}")
    df = pd.read_csv(path)
    variants = sorted(df["variant"].unique().tolist())
    if variants != ["augmented_action", "unseen"]:
        raise AssertionError(f"Expected parallel variants augmented_action/unseen, got {variants}")
    print("parallel append check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=f"/tmp/fetch_delay_pipeline_smoke_{int(time.time())}")
    parser.add_argument("--keep-dir", action="store_true")
    parser.add_argument("--skip-shape-check", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    print(f"Smoke work dir: {work_dir}")

    try:
        run(
            [
                sys.executable,
                "-m",
                "py_compile",
                "fetch_delay_experiments.py",
                "plot_fetch_delay_experiments.py",
                "check_fetch_delay_results.py",
                "run_fetch_delay_pipeline.py",
                "wrappers_rd.py",
            ]
        )
        run(
            [
                sys.executable,
                "fetch_delay_experiments.py",
                "--dry-run",
                "--output-dir",
                work_dir / "dry_run",
                "--shard-count",
                "4",
                "--shard-index",
                "0",
            ]
        )
        dry_manifest = work_dir / "dry_run" / "run_manifest_shard0-of-4.json"
        if not dry_manifest.exists():
            raise AssertionError(f"Dry-run manifest was not written: {dry_manifest}")

        if not args.skip_shape_check:
            check_generalization_shapes()

        smoke_dir = work_dir / "main"
        run(
            [
                sys.executable,
                "fetch_delay_experiments.py",
                "--output-dir",
                smoke_dir,
                "--results-csv",
                smoke_dir / "evaluations.csv",
                "--experiments",
                "state_constant",
                "--variants",
                "unseen",
                "augmented_action",
                "augmented_action_delay",
                "--seeds",
                "0",
                "--state-constant-delay",
                "0",
                "--steps",
                "5",
                "--n-eval-episodes",
                "1",
                "--device",
                "cpu",
                "--buffer-size",
                "1000",
                "--batch-size",
                "16",
                "--learning-starts",
                "1",
            ]
        )

        resume_dir = work_dir / "resume"
        run(
            [
                sys.executable,
                "fetch_delay_experiments.py",
                "--output-dir",
                resume_dir,
                "--results-csv",
                resume_dir / "evaluations.csv",
                "--experiments",
                "state_constant",
                "--variants",
                "unseen",
                "--seeds",
                "0",
                "--state-constant-delay",
                "0",
                "--steps",
                "5",
                "--n-eval-episodes",
                "1",
                "--device",
                "cpu",
                "--buffer-size",
                "1000",
                "--batch-size",
                "16",
                "--learning-starts",
                "1",
            ]
        )
        run(
            [
                sys.executable,
                "fetch_delay_experiments.py",
                "--output-dir",
                resume_dir,
                "--results-csv",
                resume_dir / "evaluations.csv",
                "--experiments",
                "state_constant",
                "--variants",
                "unseen",
                "--seeds",
                "0",
                "--state-constant-delay",
                "0",
                "--n-eval-episodes",
                "3",
                "--device",
                "cpu",
                "--eval-only",
            ]
        )
        check_resume_csv(resume_dir / "evaluations.csv")

        parallel_csv = work_dir / "parallel" / "evaluations.csv"
        parallel_csv.parent.mkdir(parents=True)
        cmd_unseen = [
            sys.executable,
            "fetch_delay_experiments.py",
            "--output-dir",
            smoke_dir,
            "--results-csv",
            parallel_csv,
            "--experiments",
            "state_constant",
            "--variants",
            "unseen",
            "--seeds",
            "0",
            "--state-constant-delay",
            "0",
            "--n-eval-episodes",
            "1",
            "--device",
            "cpu",
            "--eval-only",
        ]
        cmd_aug = [
            sys.executable,
            "fetch_delay_experiments.py",
            "--output-dir",
            smoke_dir,
            "--results-csv",
            parallel_csv,
            "--experiments",
            "state_constant",
            "--variants",
            "augmented_action",
            "--seeds",
            "0",
            "--state-constant-delay",
            "0",
            "--n-eval-episodes",
            "1",
            "--device",
            "cpu",
            "--eval-only",
        ]
        print("\n$ " + " ".join(str(part) for part in cmd_unseen) + "  &", flush=True)
        print("$ " + " ".join(str(part) for part in cmd_aug) + "  &", flush=True)
        p1 = subprocess.Popen(cmd_unseen, cwd=ROOT)
        p2 = subprocess.Popen(cmd_aug, cwd=ROOT)
        if p1.wait() != 0 or p2.wait() != 0:
            raise subprocess.CalledProcessError(1, "parallel eval workers")
        check_parallel_csv(parallel_csv)

        eval_copy = smoke_dir / "evaluations_copy.csv"
        shutil.copy2(smoke_dir / "evaluations.csv", eval_copy)
        plots_dir = work_dir / "plots"
        run(
            [
                sys.executable,
                "plot_fetch_delay_experiments.py",
                "--results-csv",
                smoke_dir / "evaluations.csv",
                eval_copy,
                "--output-dir",
                plots_dir,
                "--prefix",
                "FetchPush_smoke",
            ]
        )
        run(
            [
                sys.executable,
                "check_fetch_delay_results.py",
                "--results-csv",
                smoke_dir / "evaluations.csv",
                eval_copy,
                "--manifest",
                smoke_dir / "run_manifest.json",
                "--plots-dir",
                plots_dir,
                "--prefix",
                "FetchPush_smoke",
                "--experiments",
                "state_constant",
                "--variants",
                "unseen",
                "augmented_action",
                "augmented_action_delay",
                "--seeds",
                "0",
                "--state-constant-delay",
                "0",
                "--n-eval-episodes",
                "1",
            ]
        )
    finally:
        if args.keep_dir:
            print(f"Kept smoke work dir: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    print("\nFetch delay pipeline smoke test passed")


if __name__ == "__main__":
    main()
