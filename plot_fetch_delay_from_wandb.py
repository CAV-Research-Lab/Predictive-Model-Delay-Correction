"""
Build interim Fetch delay plots directly from completed W&B evaluation runs.

This is intended for previewing a partially completed Slurm sweep before the
cluster-side finalize step has produced the local evaluations CSV.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from plot_fetch_delay_experiments import configure_style, plot_three_figures, write_plot_summary


DEFAULT_PROJECT = "cavlab/fetch-delay"
DEFAULT_GROUP = "FetchPush-RemotePDNorm-v0-80000steps-20evaleps-0_1_2_3_4"

CONFIG_COLUMNS = [
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
    "model_path",
]
NUMERIC_COLUMNS = [
    "episode_steps",
    "episode_return",
    "episode_mean_reward",
    "episode_mean_distance",
    "episode_final_goal_distance",
    "episode_min_goal_distance",
    "episode_success",
]
SUMMARY_MEAN_KEYS = {
    "episode_steps": "eval/episode_steps_mean",
    "episode_return": "eval/episode_return_mean",
    "episode_mean_reward": "eval/episode_mean_reward_mean",
    "episode_mean_distance": "eval/episode_mean_distance_mean",
    "episode_final_goal_distance": "eval/episode_final_goal_distance_mean",
    "episode_min_goal_distance": "eval/episode_min_goal_distance_mean",
    "episode_success": "eval/success_rate",
}
HISTORY_KEYS = ["episode", *(f"eval/{column}" for column in NUMERIC_COLUMNS)]
METRICS = [
    "episode_return",
    "episode_mean_reward",
    "episode_mean_distance",
    "episode_final_goal_distance",
    "episode_min_goal_distance",
    "episode_success",
]


def import_wandb():
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in this interpreter. Install it with:\n"
            "  uv --cache-dir /tmp/uv-cache pip install --python .venv/bin/python wandb"
        ) from exc
    return wandb


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_finished(run) -> bool:
    return str(getattr(run, "state", "")).lower() == "finished"


def is_evaluation_run(run) -> bool:
    if getattr(run, "job_type", None) == "evaluation":
        return True
    config = getattr(run, "config", {}) or {}
    summary = getattr(run, "summary", {}) or {}
    return bool(config.get("experiment")) and summary.get("eval/episode_count") is not None


def run_base_row(run) -> dict[str, Any]:
    config = getattr(run, "config", {}) or {}
    row = {column: config.get(column) for column in CONFIG_COLUMNS}
    row["seed"] = safe_int(row.get("seed"))
    row["x_value"] = safe_float(row.get("x_value"))
    row["wandb_run_id"] = run.id
    row["wandb_run_name"] = run.name
    row["wandb_url"] = getattr(run, "url", None)
    row["wandb_group"] = getattr(run, "group", None)
    return row


def rows_from_history(run) -> list[dict[str, Any]]:
    base = run_base_row(run)
    rows: list[dict[str, Any]] = []
    for item in run.scan_history(keys=HISTORY_KEYS, page_size=1000):
        numeric = {
            column: safe_float(item.get(f"eval/{column}"))
            for column in NUMERIC_COLUMNS
        }
        if all(value is None for value in numeric.values()):
            continue
        episode = safe_int(item.get("episode", item.get("_step")))
        row = {**base, **numeric, "episode": episode if episode is not None else len(rows)}
        rows.append(row)
    return rows


def row_from_summary(run) -> dict[str, Any] | None:
    base = run_base_row(run)
    summary = getattr(run, "summary", {}) or {}
    numeric = {
        column: safe_float(summary.get(summary_key))
        for column, summary_key in SUMMARY_MEAN_KEYS.items()
    }
    if all(value is None for value in numeric.values()):
        return None
    return {**base, **numeric, "episode": 0, "wandb_row_source": "summary"}


def rows_from_run(run, source: str) -> tuple[list[dict[str, Any]], str]:
    if source in {"history", "auto"}:
        rows = rows_from_history(run)
        if rows:
            for row in rows:
                row["wandb_row_source"] = "history"
            return rows, "history"

    if source in {"summary", "auto", "history"}:
        summary_row = row_from_summary(run)
        if summary_row is not None:
            return [summary_row], "summary"

    return [], "missing"


def iter_groups(groups):
    if groups is None:
        return []
    if isinstance(groups, str):
        return [groups]
    return groups


def iter_candidate_runs(args):
    wandb = import_wandb()
    api = wandb.Api(timeout=args.timeout)
    for group in iter_groups(args.group):
        filters: dict[str, Any] = {}
        if group:
            filters["group"] = group
        if not args.include_running:
            filters["state"] = "finished"

        job_filters = {**filters, "jobType": "evaluation"}
        print(
            f"Querying W&B runs: project={args.project} group={group!r} "
            f"row_source={args.row_source}",
            flush=True,
        )
        yielded = 0
        for run in api.runs(args.project, filters=job_filters, per_page=50):
            yielded += 1
            yield run

        if yielded:
            continue

        print("No runs returned by the jobType filter; retrying with group/state filters only.", flush=True)
        for run in api.runs(args.project, filters=filters, per_page=50):
            yield run


def build_preview_dataframe(args) -> tuple[pd.DataFrame, Counter[str], list[Any]]:
    source_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    runs: list[Any] = []
    for index, run in enumerate(iter_candidate_runs(args), start=1):
        if args.limit and len(runs) >= args.limit:
            break
        if not args.include_running and not is_finished(run):
            continue
        if not is_evaluation_run(run):
            continue

        runs.append(run)
        run_rows, source = rows_from_run(run, args.row_source)
        source_counts[source] += 1
        rows.extend(run_rows)
        if len(runs) % 25 == 0:
            print(f"Fetched {len(runs)} evaluation runs", flush=True)

    if not rows:
        raise SystemExit("No plottable evaluation rows were found in W&B.")

    df = pd.DataFrame(rows)
    for column in ["x_value", *NUMERIC_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    return df, source_counts, runs


def write_coverage_report(df: pd.DataFrame, source_counts: Counter[str], runs: list[Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"evaluation_runs: {len(runs)}",
        f"episode_rows: {len(df)}",
        f"row_sources: {dict(source_counts)}",
    ]
    for column in ["experiment", "variant", "seed"]:
        if column in df.columns:
            counts = Counter(str(value) for value in df[column].dropna().tolist())
            lines.append(f"{column}_counts: {dict(sorted(counts.items()))}")
    if "wandb_group" in df.columns:
        counts = Counter(str(value) for value in df["wandb_group"].dropna().tolist())
        lines.append(f"group_counts: {dict(sorted(counts.items()))}")
    latest = max((getattr(run, "created_at", None) for run in runs), default=None)
    if latest:
        lines.append(f"latest_run_created_at: {latest}")
    output.write_text("\n".join(lines) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B path, usually entity/project.")
    parser.add_argument("--group", nargs="+", default=[DEFAULT_GROUP])
    parser.add_argument("--output-dir", default="fetch_delay_runs/wandb_preview_plots")
    parser.add_argument("--prefix", default="FetchPush_wandb_partial")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--coverage-report", default=None)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["episode_return"])
    parser.add_argument("--row-source", choices=["history", "summary", "auto"], default="history")
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    return parser


def metric_prefix(prefix: str, metric: str, metric_count: int) -> str:
    if metric_count == 1:
        return prefix
    return f"{prefix}_{metric.removeprefix('episode_')}"


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, source_counts, runs = build_preview_dataframe(args)
    results_csv = Path(args.results_csv) if args.results_csv else output_dir / f"{args.prefix}_wandb_rows.csv"
    df.to_csv(results_csv, index=False)

    coverage_report = (
        Path(args.coverage_report)
        if args.coverage_report
        else output_dir / f"{args.prefix}_coverage.txt"
    )
    coverage_report = write_coverage_report(df, source_counts, runs, coverage_report)

    print(f"Saved W&B-derived rows: {results_csv}")
    print(f"Saved coverage report: {coverage_report}")
    print("Saved plots:")
    for metric in args.metrics:
        prefix = metric_prefix(args.prefix, metric, len(args.metrics))
        outputs = plot_three_figures(df, output_dir, prefix, metric)
        summary_csv = (
            Path(args.summary_csv)
            if args.summary_csv and len(args.metrics) == 1
            else output_dir / f"{prefix}_plot_summary.csv"
        )
        summary_csv = write_plot_summary(df, metric, summary_csv)
        for output in outputs:
            print(f"  {output}")
        print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
