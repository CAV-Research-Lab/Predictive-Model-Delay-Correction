"""
Plot FetchPush delayed-RL experiment results with IQM/IQR summaries.
"""

import argparse
import glob
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = ["unseen", "augmented_action", "augmented_action_delay"]
VARIANT_LABELS = {
    "unseen": "Unseen delay",
    "augmented_action": "Augmented (+action buffer)",
    "augmented_action_delay": "Augmented (+action buffer + delay values)",
}
COLORS = {
    "unseen": "#2f5f8f",
    "augmented_action": "#b14e2c",
    "augmented_action_delay": "#3f7f4f",
}
CONTROL_MODE_ORDER = ["pd_gain", "direct_control"]
CONTROL_MODE_LABELS = {
    "pd_gain": "PD gain control",
    "direct_control": "Direct action control",
}
CONTROL_MODE_COLORS = {
    "pd_gain": "#2f5f8f",
    "direct_control": "#b14e2c",
}


def import_wandb():
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in this interpreter. Install it with:\n"
            "  uv --cache-dir /tmp/uv-cache pip install --python .venv/bin/python wandb"
        ) from exc
    return wandb


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
    lo, hi = np.quantile(values, [0.25, 0.75])
    middle = values[(values >= lo) & (values <= hi)]
    if len(middle) == 0:
        return float(np.mean(values))
    return float(np.mean(middle))


def aggregate(df, group_cols, metric):
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        q1, q3 = (np.nan, np.nan) if len(finite) == 0 else np.quantile(finite, [0.25, 0.75])
        row = dict(zip(group_cols, keys))
        row.update({"iqm": iqm(finite), "q1": q1, "q3": q3, "n": len(finite)})
        rows.append(row)
    return pd.DataFrame(rows)


def write_plot_summary(df, metric, output):
    group_cols = [
        "env_id",
        "experiment",
        "comparison",
        "variant",
        "variant_label",
        "series",
        "train_delay",
        "eval_delay",
        "x_value",
        "x_label",
    ]
    present_cols = [col for col in group_cols if col in df.columns]
    summary = aggregate(df, present_cols, metric)
    summary.insert(0, "metric", metric)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return output


def metric_label(metric):
    labels = {
        "episode_return": "IQM episodic return",
        "episode_mean_reward": "IQM mean reward",
        "episode_mean_distance": "IQM mean tracking distance",
        "episode_final_goal_distance": "IQM final goal distance",
        "episode_min_goal_distance": "IQM minimum goal distance",
        "episode_success": "IQM success rate",
    }
    return labels.get(metric, f"IQM {metric}")


def control_mode_from_env_id(env_id):
    env_id = str(env_id)
    if "RemoteDirect" in env_id:
        return "direct_control"
    if "RemotePDNorm" in env_id:
        return "pd_gain"
    return env_id


def expand_result_paths(results_csv):
    paths = []
    for item in results_csv:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing results CSV file(s): {missing_text}")
    return paths


def prepare_df(results_csv, metric):
    paths = expand_result_paths(results_csv)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if metric not in df.columns:
        raise KeyError(f"Metric {metric!r} not found in result CSVs")
    key_cols = [
        "env_id",
        "experiment",
        "comparison",
        "variant",
        "train_delay",
        "eval_delay",
        "seed",
        "episode",
    ]
    present_key_cols = [col for col in key_cols if col in df.columns]
    if present_key_cols:
        df = df.drop_duplicates(subset=present_key_cols, keep="last")
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def plot_sweep(ax, df, metric, title, xlabel):
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    summary = aggregate(df, ["variant", "x_value"], metric)
    for variant in VARIANT_ORDER:
        part = summary[summary["variant"] == variant].sort_values("x_value")
        if part.empty:
            continue
        xs = part["x_value"].to_numpy(dtype=float)
        ys = part["iqm"].to_numpy(dtype=float)
        q1 = part["q1"].to_numpy(dtype=float)
        q3 = part["q3"].to_numpy(dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=COLORS[variant], label=VARIANT_LABELS[variant])
        ax.fill_between(xs, q1, q3, color=COLORS[variant], alpha=0.18, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label(metric))


def plot_state_bars(ax, df, metric, title):
    ax.set_title(title)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    summary = aggregate(df, ["variant"], metric)
    xs = np.arange(len(VARIANT_ORDER))
    heights = []
    lower = []
    upper = []
    colors = []
    for variant in VARIANT_ORDER:
        row = summary[summary["variant"] == variant]
        if row.empty:
            heights.append(np.nan)
            lower.append(0)
            upper.append(0)
        else:
            item = row.iloc[0]
            heights.append(item["iqm"])
            lower.append(max(0.0, item["iqm"] - item["q1"]))
            upper.append(max(0.0, item["q3"] - item["iqm"]))
        colors.append(COLORS[variant])

    ax.bar(xs, heights, color=colors, width=0.72)
    ax.errorbar(xs, heights, yerr=[lower, upper], fmt="none", color="black", capsize=4, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Unseen", "+ actions", "+ actions\n+ delays"])
    ax.set_ylabel(metric_label(metric))


def plot_action_vs_observation(ax, df, metric):
    ax.set_title("Action vs Observation Delay")
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    summary = aggregate(df, ["variant", "comparison", "x_value"], metric)
    linestyles = {"action_delay_only": "-", "observation_delay_only": "--"}
    comparison_labels = {"action_delay_only": "action", "observation_delay_only": "observation"}
    for variant in VARIANT_ORDER:
        for comparison, linestyle in linestyles.items():
            part = summary[(summary["variant"] == variant) & (summary["comparison"] == comparison)].sort_values("x_value")
            if part.empty:
                continue
            xs = part["x_value"].to_numpy(dtype=float)
            ys = part["iqm"].to_numpy(dtype=float)
            q1 = part["q1"].to_numpy(dtype=float)
            q3 = part["q3"].to_numpy(dtype=float)
            label = f"{VARIANT_LABELS[variant]}: {comparison_labels[comparison]}"
            ax.plot(xs, ys, marker="o", linewidth=1.6, linestyle=linestyle, color=COLORS[variant], label=label)
            ax.fill_between(xs, q1, q3, color=COLORS[variant], alpha=0.12, linewidth=0)

    ax.set_xlabel("Delay length")
    ax.set_ylabel(metric_label(metric))


def plot_control_mode_comparison(df, output_dir, prefix, metric):
    df = df[df["experiment"] == "action_vs_observation"].copy()
    if df.empty or "env_id" not in df.columns:
        return None

    df["control_mode"] = df["env_id"].map(control_mode_from_env_id)
    present_modes = set(df["control_mode"].dropna().unique())
    if not {"pd_gain", "direct_control"}.issubset(present_modes):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
    linestyles = {"action_delay_only": "-", "observation_delay_only": "--"}
    comparison_labels = {"action_delay_only": "action delay", "observation_delay_only": "observation delay"}

    for ax, variant in zip(axes, VARIANT_ORDER):
        part_df = df[df["variant"] == variant]
        ax.set_title(VARIANT_LABELS[variant])
        if part_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        summary = aggregate(part_df, ["control_mode", "comparison", "x_value"], metric)
        for control_mode in CONTROL_MODE_ORDER:
            for comparison, linestyle in linestyles.items():
                part = summary[
                    (summary["control_mode"] == control_mode) & (summary["comparison"] == comparison)
                ].sort_values("x_value")
                if part.empty:
                    continue
                xs = part["x_value"].to_numpy(dtype=float)
                ys = part["iqm"].to_numpy(dtype=float)
                q1 = part["q1"].to_numpy(dtype=float)
                q3 = part["q3"].to_numpy(dtype=float)
                label = f"{CONTROL_MODE_LABELS[control_mode]}: {comparison_labels[comparison]}"
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.7,
                    linestyle=linestyle,
                    color=CONTROL_MODE_COLORS[control_mode],
                    label=label,
                )
                ax.fill_between(xs, q1, q3, color=CONTROL_MODE_COLORS[control_mode], alpha=0.13, linewidth=0)

        ax.set_xlabel("Delay length")
        ax.set_ylabel(metric_label(metric))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))

    output = Path(output_dir) / f"{prefix}_control_mode_comparison.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_three_figures(df, output_dir, prefix, metric):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    plot_sweep(
        axes[0],
        df[df["experiment"] == "constant_sweep"],
        metric,
        "Constant Delay Length Impact",
        "Delay pair (a,o) with a=o",
    )
    plot_sweep(
        axes[1],
        df[df["experiment"] == "stochastic_sweep"],
        metric,
        "Stochastic Observation Delay Impact",
        "Observation-delay upper bound",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(output_dir / f"{prefix}_delay_length_impact.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    plot_state_bars(axes[0], df[df["experiment"] == "state_constant"], metric, "State Information: Constant Delay")
    plot_state_bars(axes[1], df[df["experiment"] == "state_stochastic"], metric, "State Information: Stochastic Delay")
    fig.savefig(output_dir / f"{prefix}_state_information.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
    plot_action_vs_observation(axes[0], df[df["experiment"] == "action_vs_observation"], metric)
    plot_sweep(
        axes[1],
        df[df["experiment"] == "generalization_constant"],
        metric,
        "Train Constant (10,10)",
        "Evaluation delay pair",
    )
    plot_sweep(
        axes[2],
        df[df["experiment"] == "generalization_stochastic"],
        metric,
        "Train Stochastic (0,0-10)",
        "Evaluation observation upper bound",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    fig.savefig(output_dir / f"{prefix}_delay_structure_generalization.png", bbox_inches="tight")
    plt.close(fig)

    outputs = [
        output_dir / f"{prefix}_delay_length_impact.png",
        output_dir / f"{prefix}_state_information.png",
        output_dir / f"{prefix}_delay_structure_generalization.png",
    ]
    control_output = plot_control_mode_comparison(df, output_dir, prefix, metric)
    if control_output is not None:
        outputs.append(control_output)
    return outputs


def default_wandb_run_id(args):
    raw = "|".join(
        [
            args.prefix,
            args.metric,
            str(args.output_dir),
            "|".join(str(path) for path in args.results_csv),
            str(args.wandb_group),
        ]
    )
    return f"fetch-delay-plots-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def default_wandb_run_name(args):
    name = f"{args.prefix} aggregate plots ({args.metric})"
    return f"{args.wandb_run_name_prefix} | {name}" if args.wandb_run_name_prefix else name


def log_plots_to_wandb(args, df, outputs, summary_csv):
    if not args.wandb:
        return

    wandb = import_wandb()
    summary_df = pd.read_csv(summary_csv)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type="aggregate-plots",
        id=args.wandb_run_id or default_wandb_run_id(args),
        name=args.wandb_name or default_wandb_run_name(args),
        config={
            "prefix": args.prefix,
            "metric": args.metric,
            "results_csv": [str(path) for path in args.results_csv],
            "output_dir": str(args.output_dir),
            "row_count": int(len(df)),
            "env_ids": sorted(df["env_id"].dropna().unique().tolist()) if "env_id" in df else [],
            "experiments": sorted(df["experiment"].dropna().unique().tolist()) if "experiment" in df else [],
            "variants": sorted(df["variant"].dropna().unique().tolist()) if "variant" in df else [],
        },
        resume="allow",
        mode=args.wandb_mode,
        reinit="finish_previous",
    )

    payload = {
        "plots/delay_length_impact": wandb.Image(str(outputs[0])),
        "plots/state_information": wandb.Image(str(outputs[1])),
        "plots/delay_structure_generalization": wandb.Image(str(outputs[2])),
        "plots/iqm_iqr_summary": wandb.Table(dataframe=summary_df),
    }
    if len(outputs) > 3:
        payload["plots/control_mode_comparison"] = wandb.Image(str(outputs[3]))
    run.log(payload)
    run.summary["plot_count"] = len(outputs)
    run.summary["summary_rows"] = len(summary_df)
    run.summary["summary_csv"] = str(summary_csv)
    for output in outputs:
        run.summary[output.stem] = str(output)
    run.finish()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", nargs="+", default=["fetch_delay_runs/evaluations.csv"])
    parser.add_argument("--output-dir", default="fetch_delay_runs/plots")
    parser.add_argument("--prefix", default="FetchPush")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument(
        "--metric",
        default="episode_return",
        choices=[
            "episode_return",
            "episode_mean_reward",
            "episode_mean_distance",
            "episode_final_goal_distance",
            "episode_min_goal_distance",
            "episode_success",
        ],
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fetch-delay")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name-prefix", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser


def main():
    args = build_parser().parse_args()
    configure_style()
    df = prepare_df(args.results_csv, args.metric)
    outputs = plot_three_figures(df, args.output_dir, args.prefix, args.metric)
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = Path(args.output_dir) / f"{args.prefix}_plot_summary.csv"
    summary_csv = write_plot_summary(df, args.metric, summary_csv)
    log_plots_to_wandb(args, df, outputs, summary_csv)
    print("Saved plots:")
    for path in outputs:
        print(f"  {path}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
