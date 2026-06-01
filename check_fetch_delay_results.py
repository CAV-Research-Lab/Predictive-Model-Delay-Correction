"""
Validate FetchPush delay experiment coverage and plot outputs.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd

from fetch_delay_experiments import EXPERIMENTS, VARIANTS, build_run_specs


PLOT_NAMES = (
    "{prefix}_delay_length_impact.png",
    "{prefix}_state_information.png",
    "{prefix}_delay_structure_generalization.png",
)
SUMMARY_NAME = "{prefix}_plot_summary.csv"


def expand_paths(items):
    paths = []
    for item in items:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))
    return paths


def load_results(paths):
    frames = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)

    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing results CSV file(s): {missing_text}")
    if not frames:
        raise FileNotFoundError("No result CSVs matched")

    df = pd.concat(frames, ignore_index=True)
    key_cols = ["env_id", "experiment", "comparison", "variant", "train_delay", "eval_delay", "seed", "episode"]
    return df.drop_duplicates(subset=key_cols, keep="last")


def load_manifests(paths):
    manifests = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        with path.open("r") as handle:
            manifest = json.load(handle)
        manifest["source_manifest"] = str(path)
        manifests.append(manifest)

    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing manifest file(s): {missing_text}")
    if not manifests:
        raise FileNotFoundError("No manifest files matched")
    return manifests


def expected_eval_rows(args):
    specs = build_run_specs(args)
    rows = set()
    for seed in args.seeds:
        for spec in specs:
            for eval_spec in spec.eval_specs:
                for episode in range(args.n_eval_episodes):
                    rows.add(
                        (
                            args.env_id,
                            eval_spec.experiment,
                            eval_spec.comparison,
                            spec.variant,
                            spec.train_setting.display,
                            eval_spec.setting.display,
                            seed,
                            episode,
                        )
                    )
    return rows


def expected_eval_rows_from_manifests(manifests):
    rows = set()
    for manifest in manifests:
        env_id = manifest["env_id"]
        n_eval_episodes = int(manifest["n_eval_episodes"])
        seeds = [int(seed) for seed in manifest["seeds"]]
        for seed in seeds:
            for spec in manifest["specs"]:
                for eval_spec in spec["eval_specs"]:
                    for episode in range(n_eval_episodes):
                        rows.add(
                            (
                                env_id,
                                eval_spec["experiment"],
                                eval_spec["comparison"],
                                spec["variant"],
                                spec["train_delay"]["display"],
                                eval_spec["eval_delay"]["display"],
                                seed,
                                episode,
                            )
                        )
    return rows


def observed_eval_rows(df):
    required = ["env_id", "experiment", "comparison", "variant", "train_delay", "eval_delay", "seed", "episode"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Result CSV missing required columns: {', '.join(missing_cols)}")

    rows = set()
    for row in df[required].itertuples(index=False):
        rows.add(
            (
                row.env_id,
                row.experiment,
                row.comparison,
                row.variant,
                row.train_delay,
                row.eval_delay,
                int(row.seed),
                int(row.episode),
            )
        )
    return rows


def validate_manifest_coverage(manifests):
    spec_keys = set()
    duplicate_specs = []
    for manifest in manifests:
        for spec in manifest["specs"]:
            key = (
                manifest["env_id"],
                spec["variant"],
                spec["train_delay"]["name"],
                int(spec["act_buffer_len"]),
                int(spec["obs_buffer_len"]),
            )
            if key in spec_keys:
                duplicate_specs.append((manifest["source_manifest"], key))
            spec_keys.add(key)

    spec_count = len(spec_keys)
    eval_jobs = sum(int(manifest["eval_jobs"]) for manifest in manifests)
    train_jobs = sum(int(manifest["train_jobs"]) for manifest in manifests)
    all_spec_counts = {int(manifest["all_spec_count"]) for manifest in manifests}
    if len(all_spec_counts) > 1:
        raise ValueError(f"Manifest all_spec_count values disagree: {sorted(all_spec_counts)}")
    all_spec_count = next(iter(all_spec_counts)) if all_spec_counts else 0
    return {
        "manifest_count": len(manifests),
        "spec_count": spec_count,
        "all_spec_count": all_spec_count,
        "train_jobs": train_jobs,
        "eval_jobs": eval_jobs,
        "duplicate_specs": duplicate_specs,
        "complete": spec_count == all_spec_count,
    }


def validate_plots(plots_dir, prefix):
    plots_dir = Path(plots_dir)
    missing = []
    for name in PLOT_NAMES:
        path = plots_dir / name.format(prefix=prefix)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
    return missing


def validate_summary(summary_csv):
    path = Path(summary_csv)
    if not path.exists() or path.stat().st_size == 0:
        return [path], pd.DataFrame()
    df = pd.read_csv(path)
    required = {"metric", "env_id", "experiment", "variant", "iqm", "q1", "q3", "n"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise KeyError(f"Summary CSV missing required columns: {', '.join(missing_cols)}")
    return [], df


def expected_summary_rows(expected_eval_rows):
    return {
        (env_id, experiment, comparison, variant, train_delay, eval_delay)
        for env_id, experiment, comparison, variant, train_delay, eval_delay, _seed, _episode in expected_eval_rows
    }


def observed_summary_rows(summary_df):
    required = ["env_id", "experiment", "comparison", "variant", "train_delay", "eval_delay"]
    missing_cols = [col for col in required if col not in summary_df.columns]
    if missing_cols:
        raise KeyError(f"Summary CSV missing required columns: {', '.join(missing_cols)}")
    rows = set()
    for row in summary_df[required].itertuples(index=False):
        rows.add((row.env_id, row.experiment, row.comparison, row.variant, row.train_delay, row.eval_delay))
    return rows


def validate_summary_values(summary_df):
    bad_rows = []
    for idx, row in summary_df.iterrows():
        try:
            iqm_value = float(row["iqm"])
            q1_value = float(row["q1"])
            q3_value = float(row["q3"])
            n_value = int(row["n"])
        except (KeyError, TypeError, ValueError):
            bad_rows.append((idx, "non-numeric"))
            continue
        if not (math.isfinite(iqm_value) and math.isfinite(q1_value) and math.isfinite(q3_value)):
            bad_rows.append((idx, "non-finite"))
            continue
        if n_value <= 0:
            bad_rows.append((idx, "n<=0"))
            continue
        if not (q1_value <= iqm_value <= q3_value):
            bad_rows.append((idx, "iqm outside IQR"))
    return bad_rows


def validate_models(df):
    if "model_path" not in df.columns:
        raise KeyError("Result CSV missing required column: model_path")
    paths = sorted({Path(path) for path in df["model_path"].dropna().astype(str) if path})
    missing = [path for path in paths if not path.exists() or path.stat().st_size == 0]
    return missing


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    parser.add_argument("--results-csv", nargs="+", default=["fetch_delay_runs/evaluations.csv"])
    parser.add_argument("--manifest", nargs="+", default=None)
    parser.add_argument("--plots-dir", default="fetch_delay_runs/plots")
    parser.add_argument("--prefix", default="FetchPush")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-eval-episodes", type=int, default=10)
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
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--allow-partial-manifest", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    df = load_results(expand_paths(args.results_csv))
    manifests = None
    if args.manifest:
        manifests = load_manifests(expand_paths(args.manifest))
        manifest_summary = validate_manifest_coverage(manifests)
        expected = expected_eval_rows_from_manifests(manifests)
        print(
            "Manifest files: "
            f"{manifest_summary['manifest_count']} "
            f"(specs {manifest_summary['spec_count']}/{manifest_summary['all_spec_count']}, "
            f"train jobs {manifest_summary['train_jobs']}, eval jobs {manifest_summary['eval_jobs']})"
        )
        if manifest_summary["duplicate_specs"]:
            print("\nDuplicate manifest specs:")
            for source, key in manifest_summary["duplicate_specs"][:20]:
                print(f"  {source}: {key}")
            if len(manifest_summary["duplicate_specs"]) > 20:
                print(f"  ... and {len(manifest_summary['duplicate_specs']) - 20} more")
        if not manifest_summary["complete"]:
            print(
                "\nPartial manifest coverage: "
                f"{manifest_summary['spec_count']} of {manifest_summary['all_spec_count']} specs"
            )
    else:
        manifest_summary = None
        expected = expected_eval_rows(args)
    observed = observed_eval_rows(df)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    print(f"Expected rows: {len(expected)}")
    print(f"Observed rows: {len(observed)}")
    print(f"Missing rows:  {len(missing)}")
    print(f"Extra rows:    {len(extra)}")

    if missing:
        print("\nFirst missing rows:")
        for row in missing[:20]:
            print("  ", row)

    if extra:
        print("\nFirst extra rows:")
        for row in extra[:20]:
            print("  ", row)

    missing_plots = []
    missing_summary = []
    summary_missing_rows = []
    summary_extra_rows = []
    bad_summary_values = []
    missing_models = []
    if not args.skip_plots:
        missing_plots = validate_plots(args.plots_dir, args.prefix)
        if missing_plots:
            print("\nMissing plot files:")
            for path in missing_plots:
                print(f"  {path}")
        else:
            print("Plot files: present")

    if not args.skip_summary:
        summary_csv = args.summary_csv
        if summary_csv is None:
            summary_csv = Path(args.plots_dir) / SUMMARY_NAME.format(prefix=args.prefix)
        missing_summary, summary_df = validate_summary(summary_csv)
        if missing_summary:
            print("\nMissing summary files:")
            for path in missing_summary:
                print(f"  {path}")
        else:
            print("Plot summary: present")
            expected_summary = expected_summary_rows(expected)
            observed_summary = observed_summary_rows(summary_df)
            summary_missing_rows = sorted(expected_summary - observed_summary)
            summary_extra_rows = sorted(observed_summary - expected_summary)
            bad_summary_values = validate_summary_values(summary_df)
            print(f"Expected summary rows: {len(expected_summary)}")
            print(f"Observed summary rows: {len(observed_summary)}")
            print(f"Missing summary rows:  {len(summary_missing_rows)}")
            print(f"Extra summary rows:    {len(summary_extra_rows)}")
            if summary_missing_rows:
                print("\nFirst missing summary rows:")
                for row in summary_missing_rows[:20]:
                    print("  ", row)
            if summary_extra_rows:
                print("\nFirst extra summary rows:")
                for row in summary_extra_rows[:20]:
                    print("  ", row)
            if bad_summary_values:
                print("\nBad summary values:")
                for idx, reason in bad_summary_values[:20]:
                    print(f"  row {idx}: {reason}")

    if not args.skip_models:
        missing_models = validate_models(df)
        if missing_models:
            print("\nMissing model files:")
            for path in missing_models[:20]:
                print(f"  {path}")
            if len(missing_models) > 20:
                print(f"  ... and {len(missing_models) - 20} more")
        else:
            print("Model files: present")

    manifest_incomplete = (
        manifest_summary is not None
        and not manifest_summary["complete"]
        and not args.allow_partial_manifest
    )
    duplicate_manifest_specs = (
        manifest_summary is not None
        and bool(manifest_summary["duplicate_specs"])
    )

    if (
        missing
        or (extra and not args.allow_extra)
        or missing_plots
        or missing_summary
        or summary_missing_rows
        or (summary_extra_rows and not args.allow_extra)
        or bad_summary_values
        or missing_models
        or manifest_incomplete
        or duplicate_manifest_specs
    ):
        raise SystemExit(1)
    print("Coverage check passed")


if __name__ == "__main__":
    main()
