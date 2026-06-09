#!/usr/bin/env python3
"""World-model state-estimation error vs delay horizon (SBSP recalibration on/off).

Chapter 3 (S3.4.4) diagnostic: turns the asserted error-propagation argument into a
measured one. The remote must estimate the true current state across the communication
delay. We measure that estimate's error as a function of the number of steps k the world
model has to bridge, k = 1..alpha (alpha = the constant action-delay horizon), comparing
the estimate to the simulator's true state in end-effector distance (cm).

Two conditions share the *same* frozen world model:
  * recalibration OFF -- naive open-loop: propagate the true state from a sync that is k
    steps stale, forward through the world model, ignoring intervening measurements. Pure
    model error compounds with k (the standard PETS/MBPO/Dreamer open-loop diagnostic).
  * recalibration ON  -- PMDC's SBSP mechanism: the k-step-ahead prediction is re-anchored
    by every arriving measurement via the constant-drift residual correction (a faithful
    replay of PMDC.recalibrate / initial_undelay; this is exactly the quantity PMDC logs
    as wm_pred_err, generalised to each k). PMDC has no observation delay -- only action
    delay -- so these measurements genuinely arrive each step.

If recalibrated error stays flat while naive open-loop error compounds, the central SBSP
claim is demonstrated rather than argued. Error is reported in end-effector distance (cm),
on the same scale as the Fetch goal tolerance eps = 5 cm, so "model error" connects
directly to "does it matter for control". Aggregated as the interquartile mean (IQM, line)
with a shaded interquartile range (IQR, 25-75th percentile) over the per-anchor samples
pooled across seeds.

The world model is trained online (PMDC's own ensemble) with PMDC_FIX_PREVOBS=1 -- the
prev_obs fix is required for the 1-step dynamics pairs (s_t, a_t)->s_{t+1} to be valid
(see PMDC_STABILITY_INVESTIGATION.md, finding F1). It is then frozen for the rollout
test. Trajectories are produced by a trained PMDC policy so the test distribution is the
deployed one; the 3 seeds vary world-model initialisation/training and environment
stochasticity.

Usage:
  python3 openloop_rollout_error.py --delay-range 250-290 --seeds 0 1 2 \
      --operator-model operator_models/FetchPush-v2_SAC_seed0.zip \
      --policy pred_rl_models/PMDC/5/250-290ms/seed_0/24_step_SAC_FetchPush-RemotePDNorm-v0.zip \
      --device cuda
  python3 openloop_rollout_error.py --replot          # re-draw from saved .npz only
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# PMDC reads these env vars when the wrapper/network are constructed, so set them first.
os.environ.setdefault("PMDC_FIX_PREVOBS", "1")  # valid 1-step dynamics pairs (F1)

import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

import robo_local_remote_env  # noqa: F401,E402  registers FetchPush-RemotePDNorm-v0 etc.
from PMDC_wrapper import PMDC  # noqa: E402
from delay_correcting_training import (  # noqa: E402
    build_training_env,
    parse_delay_range,
    seed_everything,
)

REMOTE_EE = slice(0, 3)      # obs[0:3]   remote end-effector position (achieved_goal)
OPERATOR_EE = slice(11, 14)  # obs[11:14] operator end-effector position (desired_goal)
EPS_CM = 5.0                 # Fetch success tolerance: 0.05 m

CONDS = ("off", "on")  # recalibration off (naive open-loop) / on (SBSP)
COND_LABELS = {
    "off": "No recalibration (naive open-loop)",
    "on": "With SBSP recalibration",
}
COND_COLORS = {"off": "#d1495b", "on": "#1f77b4"}
METRICS = ("remote_ee", "operator_ee", "track", "full_l2")
METRIC_LABELS = {
    "remote_ee": "Remote end-effector position error",
    "operator_ee": "Operator end-effector position error",
    "track": "Tracking-distance (reward) error",
    "full_l2": "Full-state L2 error",
}


def find_pmdc(env):
    """Walk the wrapper stack down to the PMDC wrapper."""
    while env is not None:
        if isinstance(env, PMDC):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError("PMDC wrapper not found in env stack")


def wm_mean_predict(pmdc, obs, action):
    """Ensemble-mean one-step prediction, exactly as PMDC.step uses it."""
    preds = [model.predict(obs, action) for model in pmdc.dc_models]
    return np.mean(preds, axis=0).astype(np.float32)


def openloop_estimates(states, actions, pmdc, alpha):
    """Naive open-loop estimate of the current state across an h-step delay (no recalibration).

    For every anchor a, roll the world model forward open-loop from the true state ``states[a]``
    using the realised action sequence; ``est[(a + h, h)]`` is the estimate of ``states[a + h]``
    obtained from a sync that is h steps stale. Error compounds with h.
    """
    est = {}
    n = len(states)
    for a in range(n - 1):
        cur = states[a].copy()
        for h in range(1, min(alpha, n - 1 - a) + 1):
            cur = wm_mean_predict(pmdc, cur, actions[a + h])  # action producing states[a + h]
            est[(a + h, h)] = cur.copy()
    return est


def sbsp_replay(states, actions, pmdc, h, zero_action):
    """PMDC's SBSP recalibration applied at delay h: faithful replay of PMDC.recalibrate.

    Mirrors ``PMDC.initial_undelay`` (zero-action warm start) + ``PMDC.step``/``recalibrate``
    with ``delay = h``. ``est[j]`` is PMDC's recalibrated estimate of the current true state
    ``states[j]`` -- the buffer element predicted h steps earlier and re-anchored by every
    measured residual since (the constant-drift correction). This is exactly the quantity
    PMDC logs as ``wm_pred_err`` at the delay horizon, generalised to any h. Valid for j >= h.
    """
    buf = deque()
    obs = states[0].copy()
    for _ in range(h):                              # initial_undelay: roll h steps, zero action
        obs = wm_mean_predict(pmdc, obs, zero_action)
        buf.append(obs.copy())
    future = buf[-1].copy()
    est = [None] * len(states)
    for j in range(1, len(states)):
        pred_now = buf.popleft()                    # estimate of states[j] made h steps ago
        est[j] = pred_now.copy()                    # (before applying the current residual)
        diff = states[j] - pred_now                 # measured residual -> recalibration shift
        for i in range(len(buf)):
            buf[i] = buf[i] + diff
        future = future + diff
        future = wm_mean_predict(pmdc, future, actions[j])  # extend horizon with current action
        buf.append(future.copy())
    return est


def ee_metrics(pred, true):
    """Control-relevant prediction errors (metres) between a predicted and true state."""
    rem = float(np.linalg.norm(pred[REMOTE_EE] - true[REMOTE_EE]))
    op = float(np.linalg.norm(pred[OPERATOR_EE] - true[OPERATOR_EE]))
    pred_dist = np.linalg.norm(pred[REMOTE_EE] - pred[OPERATOR_EE])
    true_dist = np.linalg.norm(true[REMOTE_EE] - true[OPERATOR_EE])
    track = float(abs(pred_dist - true_dist))
    full = float(np.linalg.norm(pred - true))
    return {"remote_ee": rem, "operator_ee": op, "track": track, "full_l2": full}


def choose_action(policy, obs, env):
    if policy is not None:
        return policy.predict(obs, deterministic=True)[0]
    return env.action_space.sample()


def run_seed(seed, delay_config, args, sink):
    """Train+freeze a world model for one seed, then accumulate rollout errors into `sink`."""
    seed_everything(seed)
    env = build_training_env(
        "PMDC", args.env_id, delay_config, seed,
        n_models=args.n_models, pretrain=args.load_wm, operator_model=args.operator_model,
    )
    pmdc = find_pmdc(env)
    pmdc.fix_prev_obs = True  # belt and braces (env var already set it at construction)
    alpha = delay_config.act_delay
    assert pmdc.delay == alpha

    policy = SAC.load(args.policy, device=args.device) if args.policy else None

    # Capture the (possibly delay-buffered) action that actually reaches PMDC.step --
    # this is the action label the world model is trained/queried with.
    captured = {"a": np.zeros_like(np.asarray(pmdc.action_space.sample(), dtype=np.float32))}
    inner_step = pmdc.step

    def recording_step(action):
        captured["a"] = np.asarray(action, dtype=np.float32).copy()
        return inner_step(action)

    pmdc.step = recording_step

    # ---- warmup: train the ensemble online until it converges and the policy/world-model
    #      loop reaches its on-distribution operating point. Skipped when --load-wm loads the
    #      policy's own persisted WM (matched pair -> on-distribution by construction). ----
    if not args.load_wm:
        pmdc.train_every = args.wm_train_every
        obs, _ = env.reset(seed=seed)
        for _ in range(args.wm_warmup_steps):
            obs, _, term, trunc, _ = env.step(choose_action(policy, obs, env))
            if term or trunc:
                obs, _ = env.reset()
        # offline polish: extra gradient steps on the (now on-distribution) replay buffer to
        # fully converge the ensemble without further (slow) env stepping.
        for _ in range(args.wm_polish_iters):
            if len(pmdc.replay_buffer) > pmdc.start_training:
                pmdc.learn()
    # freeze the world model: no more learn() in step() or reset().
    pmdc.train_every = 0
    pmdc.freeze_wm = 1  # _step_i >> 1 already, so reset() skips learn()
    final_loss, final_pred_err = pmdc.wm_loss, pmdc.wm_pred_err
    # single-sample predictions are cheaper on CPU (no per-call GPU launch overhead)
    for model in pmdc.dc_models:
        model.to("cpu")
        model.device = torch.device("cpu")

    # ---- eval: estimate the true current state across an h-step delay, with vs without
    #      recalibration. Naive open-loop (anchored at the h-stale true sync) compounds; PMDC's
    #      SBSP recalibration (re-anchored by every arriving measurement) stays bounded. ----
    zero_action = captured["a"] * 0.0
    n_anchor = 0
    track_errs = []  # true tracking error ||remote_ee - operator_ee|| (on-distribution check)
    for _ in range(args.eval_episodes):
        states, actions = [], []          # states[i] / actions[i]: true state / action at step i+1
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(choose_action(policy, obs, env))
            states.append(np.asarray(pmdc.prev_obs, dtype=np.float32).copy())  # true current state
            actions.append(captured["a"].copy())                              # action producing it
            done = term or trunc

        length = len(states)
        track_errs += [float(np.linalg.norm(s[REMOTE_EE] - s[OPERATOR_EE])) for s in states]

        naive_est = openloop_estimates(states, actions, pmdc, alpha)
        for h in range(1, alpha + 1):
            recal_est = sbsp_replay(states, actions, pmdc, h, zero_action)
            for j in range(h, length):                 # estimate of the true current state s_j
                if (j, h) not in naive_est:
                    continue
                true = states[j]
                m_off = ee_metrics(naive_est[(j, h)], true)
                m_on = ee_metrics(recal_est[j], true)
                for metric in METRICS:
                    sink[("off", metric)]["k"].append(h)
                    sink[("off", metric)]["v"].append(m_off[metric])
                    sink[("off", metric)]["seed"].append(seed)
                    sink[("on", metric)]["k"].append(h)
                    sink[("on", metric)]["v"].append(m_on[metric])
                    sink[("on", metric)]["seed"].append(seed)
                n_anchor += 1

    env.close()
    track_cm = 100.0 * float(np.mean(track_errs)) if track_errs else float("nan")
    print(f"[seed {seed}] alpha={alpha} anchors={n_anchor} "
          f"wm_loss={final_loss:.4f} wm_pred_err(@alpha)={final_pred_err:.3f} "
          f"true_tracking_err={track_cm:.1f}cm",
          flush=True)


def iqm(values):
    """Interquartile mean: mean of the central 50% (trim 25% each tail)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    k = int(np.floor(n * 0.25))
    return x[k:n - k].mean() if n - 2 * k > 0 else x.mean()


def aggregate(sink, alpha, metric, cond, seeds=None):
    """Per-horizon IQM and IQR over pooled anchors (optionally restricted to one seed)."""
    rec = sink[(cond, metric)]
    k_arr = np.asarray(rec["k"])
    v_arr = np.asarray(rec["v"]) * 100.0  # m -> cm
    s_arr = np.asarray(rec["seed"])
    horizons = np.arange(1, alpha + 1)
    line, lo, hi = (np.full(alpha, np.nan) for _ in range(3))
    for i, k in enumerate(horizons):
        mask = k_arr == k
        if seeds is not None:
            mask &= np.isin(s_arr, seeds)
        vals = v_arr[mask]
        if vals.size == 0:
            continue
        line[i] = iqm(vals)
        lo[i], hi[i] = np.percentile(vals, [25, 75])
    return horizons, line, lo, hi


def style():
    try:
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        pass


def plot_primary(sink, alpha, delay_label, out_path, metric="remote_ee", seed_list=None):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    assert isinstance(ax, Axes)
    step_ms = 10
    # Frame the y-axis on the naive (open-loop) curve so a degenerate short-bridge recal
    # transient (single-residual correction -> acceleration-sensitive at k=2) cannot blow
    # the axis and bury the main story.
    _, _, _, off_hi = aggregate(sink, alpha, metric, "off")
    ymax = float(np.nanmax(off_hi)) * 1.08 if np.isfinite(np.nanmax(off_hi)) else None
    for cond in CONDS:
        h, line, lo, hi = aggregate(sink, alpha, metric, cond)
        color = COND_COLORS[cond]
        ax.fill_between(h, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(h, line, color=color, lw=2.2, label=COND_LABELS[cond], zorder=3)
        # faint per-seed IQM lines to show the 3-seed spread
        if seed_list:
            for sd in seed_list:
                _, sl, _, _ = aggregate(sink, alpha, metric, cond, seeds=[sd])
                ax.plot(h, sl, color=color, lw=0.8, alpha=0.30, zorder=2)
    ax.axhline(EPS_CM, ls="--", color="0.35", lw=1.2,
               label=f"goal tolerance $\\epsilon$ = {EPS_CM:.0f} cm")
    ax.set_xlabel("delay bridged by world model, $k$ (env steps)")
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (cm)")
    ax.set_xlim(1, alpha)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    n_seed = len(seed_list or [])
    ax.set_title(f"World-model state-estimation error vs delay horizon\n"
                 f"FetchPush, {delay_label} ($\\alpha$={alpha} steps), IQM line + IQR band, "
                 f"{n_seed} seed{'s' if n_seed != 1 else ''}", fontsize=11)
    secax = ax.secondary_xaxis("top", functions=(lambda k: k * step_ms, lambda ms: ms / step_ms))
    secax.set_xlabel("delay bridged (ms)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_all_metrics(sink, alpha, delay_label, out_path, seed_list=None):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    for ax, metric in zip(axes.ravel(), METRICS):
        _, _, _, off_hi = aggregate(sink, alpha, metric, "off")
        ymax = float(np.nanmax(off_hi)) * 1.08 if np.isfinite(np.nanmax(off_hi)) else None
        for cond in CONDS:
            h, line, lo, hi = aggregate(sink, alpha, metric, cond)
            color = COND_COLORS[cond]
            ax.fill_between(h, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(h, line, color=color, lw=2.0, label=COND_LABELS[cond])
        if metric in ("remote_ee", "operator_ee", "track"):
            ax.axhline(EPS_CM, ls="--", color="0.35", lw=1.0)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlabel("delay bridged, $k$ (steps)")
        ax.set_ylabel("error (cm)")
        ax.set_xlim(1, alpha)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(loc="upper left", frameon=False, fontsize=8)
    n_seed = len(seed_list or [])
    fig.suptitle(f"World-model state-estimation error vs delay horizon -- FetchPush {delay_label} "
                 f"($\\alpha$={alpha}), IQM + IQR, {n_seed} seed{'s' if n_seed != 1 else ''}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def write_summary_csv(sink, alpha, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "horizon_k", "cond", "iqm_cm", "iqr_lo_cm", "iqr_hi_cm", "n"])
        for metric in METRICS:
            for cond in CONDS:
                h, line, lo, hi = aggregate(sink, alpha, metric, cond)
                rec = sink[(cond, metric)]
                k_arr = np.asarray(rec["k"])
                for i, k in enumerate(h):
                    n = int(np.sum(k_arr == k))
                    w.writerow([metric, int(k), cond,
                                f"{line[i]:.4f}", f"{lo[i]:.4f}", f"{hi[i]:.4f}", n])
    print(f"saved {out_path}")


def new_sink() -> Dict[Tuple[str, str], Dict[str, List]]:
    return {(cond, metric): {"k": [], "v": [], "seed": []}
            for cond in CONDS for metric in METRICS}


def save_npz(sink, alpha, delay_label, seed_list, path):
    flat = {"__alpha": np.array([alpha]),
            "__delay": np.array([delay_label]),
            "__seeds": np.array(seed_list)}
    for (cond, metric), rec in sink.items():
        key = f"{cond}|{metric}"
        flat[key + "|k"] = np.asarray(rec["k"], dtype=np.int16)
        flat[key + "|v"] = np.asarray(rec["v"], dtype=np.float32)
        flat[key + "|seed"] = np.asarray(rec["seed"], dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f"saved {path}")


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    alpha = int(z["__alpha"][0])
    delay_label = str(z["__delay"][0])
    seed_list = list(z["__seeds"].tolist())
    sink = new_sink()
    for cond in CONDS:
        for metric in METRICS:
            key = f"{cond}|{metric}"
            sink[(cond, metric)]["k"] = z[key + "|k"].tolist()
            sink[(cond, metric)]["v"] = z[key + "|v"].tolist()
            sink[(cond, metric)]["seed"] = z[key + "|seed"].tolist()
    return sink, alpha, delay_label, seed_list


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    p.add_argument("--delay-range", default="250-290", help="e.g. 250-290 or 90-120")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--operator-model", default="operator_models/FetchPush-v2_SAC_seed0.zip")
    p.add_argument("--policy", default="pred_rl_models/PMDC/5/250-290ms/seed_0/"
                                       "24_step_SAC_FetchPush-RemotePDNorm-v0.zip",
                   help="trained PMDC SAC policy for trajectory generation (empty = random actions)")
    p.add_argument("--wm-warmup-steps", type=int, default=80000,
                   help="online steps to train the world model before freezing (thesis horizon)")
    p.add_argument("--wm-train-every", type=int, default=4,
                   help="train the ensemble every K steps during warmup so it converges over the "
                        "full horizon (the error-propagation figure needs a converged model; "
                        "set 0 for PMDC's sparse once-per-episode deployment cadence)")
    p.add_argument("--wm-polish-iters", type=int, default=0,
                   help="extra offline ensemble gradient steps after warmup (no env stepping)")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="diagnostics/openloop_rollout")
    p.add_argument("--tag", default=None, help="optional filename tag")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="cap torch CPU threads (0 = half of os.cpu_count(); lower for parallel seeds)")
    p.add_argument("--replot", action="store_true", help="reload saved .npz and only redraw")
    p.add_argument("--combine-tags", nargs="+", default=None,
                   help="merge per-seed .npz files (these tags) under --output-dir into --tag, then plot")
    p.add_argument("--load-wm", action="store_true",
                   help="load the policy's persisted world model (pretrain=True) and skip warmup -- "
                        "use with a matched policy from train_matched_pmdc.py for on-distribution eval")
    return p


def merge_npz(out_dir, tags):
    """Concatenate per-seed .npz sinks (one per tag) into a single pooled sink."""
    sink = new_sink()
    alpha = delay_label = None
    seeds = []
    for t in tags:
        s, alpha, delay_label, sl = load_npz(out_dir / f"{t}.npz")
        for key in sink:
            for field in ("k", "v", "seed"):
                sink[key][field] += s[key][field]
        seeds += list(sl)
    return sink, alpha, delay_label, sorted(set(seeds))


def main():
    args = build_parser().parse_args()
    torch.set_num_threads(args.torch_threads or max(1, (os.cpu_count() or 2) // 2))
    delay_config = parse_delay_range(args.delay_range)
    delay_label = delay_config.label
    tag = args.tag or f"{args.env_id.split('-')[0]}_{delay_label}"
    out_dir = Path(args.output_dir)
    npz_path = out_dir / f"{tag}.npz"

    if args.combine_tags:
        sink, alpha, delay_label, seed_list = merge_npz(out_dir, args.combine_tags)
        save_npz(sink, alpha, delay_label, seed_list, npz_path)
    elif args.replot:
        sink, alpha, delay_label, seed_list = load_npz(npz_path)
    else:
        if not args.policy:
            args.policy = None
        sink = new_sink()
        seed_list = args.seeds
        for seed in seed_list:
            run_seed(seed, delay_config, args, sink)
        alpha = delay_config.act_delay
        save_npz(sink, alpha, delay_label, seed_list, npz_path)

    plot_primary(sink, alpha, delay_label, out_dir / f"{tag}_remote_ee.png",
                 metric="remote_ee", seed_list=seed_list)
    plot_all_metrics(sink, alpha, delay_label, out_dir / f"{tag}_all_metrics.png",
                     seed_list=seed_list)
    write_summary_csv(sink, alpha, out_dir / f"{tag}_summary.csv")


if __name__ == "__main__":
    main()
