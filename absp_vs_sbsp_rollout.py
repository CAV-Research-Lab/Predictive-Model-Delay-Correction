#!/usr/bin/env python3
"""ABSP vs SBSP: forward delay-horizon prediction error vs horizon (the efficiency tradeoff).

Both predictors produce the same thing -- the agent's estimate of the state k steps ahead
(the action-delay horizon) -- but differently:
  * ABSP (Action-Buffer State Prediction): re-roll the world model k steps from the latest TRUE
    observation every step, using the in-flight action buffer. O(alpha)/step, no drift assumption.
  * SBSP (State-Buffer State Prediction, PMDC default): keep a cached buffer, recalibrate it with
    the latest residual (constant-drift), extend by one. O(1)/step.

This measures, at each step t, each method's prediction for s_{t+k} (k=1..alpha) using only
information available at t, against the simulator's true s_{t+k}, in end-effector cm. Both are
faithful replays of PMDC_wrapper.py (ABSP branch / recalibrate). The thesis Ch.3 claim is that
SBSP matches ABSP's accuracy at O(1) vs O(alpha) cost -- if the curves overlap, SBSP is a free
lunch; a gap is the accuracy price of the constant-drift approximation.

Aggregated as IQM (line) + IQR (band) over per-anchor samples pooled across seeds. Same WM
training as openloop_rollout_error.py (online with PMDC_FIX_PREVOBS=1, then frozen).
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("PMDC_FIX_PREVOBS", "1")

import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

import robo_local_remote_env  # noqa: F401,E402
from delay_correcting_training import build_training_env, parse_delay_range, seed_everything  # noqa: E402
from openloop_rollout_error import (  # noqa: E402
    EPS_CM, METRICS, METRIC_LABELS, REMOTE_EE, OPERATOR_EE, aggregate, choose_action,
    ee_metrics, find_pmdc, style, wm_mean_predict,
)

CONDS = ("absp", "sbsp")
COND_LABELS = {"absp": "ABSP (re-roll from obs, $O(\\alpha)$/step)",
               "sbsp": "SBSP (recalibrate cache, $O(1)$/step)"}
COND_COLORS = {"absp": "#e08214", "sbsp": "#1f77b4"}


def absp_forward(states, actions, pmdc, t, kmax):
    """ABSP: roll the WM forward from the TRUE state s_t, returning predictions for s_{t+1..t+kmax}."""
    cur = states[t].copy()
    preds = []
    for k in range(1, kmax + 1):
        cur = wm_mean_predict(pmdc, cur, actions[t + k])  # action producing s_{t+k}
        preds.append(cur.copy())
    return preds


def sbsp_forward_buffer(states, actions, pmdc, alpha, zero_action):
    """SBSP: faithful replay of recalibrate; snaps[t] = the cached buffer at time t, i.e. the
    drift-corrected predictions [pred(t+1), ..., pred(t+alpha)] available after observing s_t."""
    buf = deque()
    obs = states[0].copy()
    for _ in range(alpha):                       # initial_undelay warm start (zero actions)
        obs = wm_mean_predict(pmdc, obs, zero_action)
        buf.append(obs.copy())
    future = buf[-1].copy()
    snaps = [None] * len(states)
    for t in range(1, len(states)):
        pred_now = buf.popleft()                 # prediction for s_t made alpha steps ago
        diff = states[t] - pred_now              # measured residual -> constant-drift shift
        for i in range(len(buf)):
            buf[i] = buf[i] + diff
        future = future + diff
        future = wm_mean_predict(pmdc, future, actions[t])
        buf.append(future.copy())
        snaps[t] = [b.copy() for b in buf]       # [pred(t+1) .. pred(t+alpha)]
    return snaps


def run_seed(seed, delay_config, args, sink):
    seed_everything(seed)
    env = build_training_env("PMDC", args.env_id, delay_config, seed,
                             n_models=args.n_models, pretrain=args.load_wm, operator_model=args.operator_model)
    pmdc = find_pmdc(env)
    pmdc.fix_prev_obs = True
    alpha = delay_config.act_delay
    policy = SAC.load(args.policy, device=args.device) if args.policy else None

    captured = {"a": np.zeros_like(np.asarray(pmdc.action_space.sample(), dtype=np.float32))}
    inner_step = pmdc.step

    def recording_step(action):
        captured["a"] = np.asarray(action, dtype=np.float32).copy()
        return inner_step(action)

    pmdc.step = recording_step

    # Skip warmup when loading the policy's own persisted WM (matched pair, on-distribution).
    if not args.load_wm:
        pmdc.train_every = args.wm_train_every
        obs, _ = env.reset(seed=seed)
        for _ in range(args.wm_warmup_steps):
            obs, _, term, trunc, _ = env.step(choose_action(policy, obs, env))
            if term or trunc:
                obs, _ = env.reset()
        for _ in range(args.wm_polish_iters):
            if len(pmdc.replay_buffer) > pmdc.start_training:
                pmdc.learn()
    pmdc.train_every = 0
    pmdc.freeze_wm = 1
    final_loss, final_pred_err = pmdc.wm_loss, pmdc.wm_pred_err
    for model in pmdc.dc_models:
        model.to("cpu")
        model.device = torch.device("cpu")

    zero_action = captured["a"] * 0.0
    n_anchor = 0
    track_errs = []
    for _ in range(args.eval_episodes):
        states, actions = [], []
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(choose_action(policy, obs, env))
            states.append(np.asarray(pmdc.prev_obs, dtype=np.float32).copy())
            actions.append(captured["a"].copy())
            done = term or trunc

        length = len(states)
        track_errs += [float(np.linalg.norm(s[REMOTE_EE] - s[OPERATOR_EE])) for s in states]
        sbsp_snaps = sbsp_forward_buffer(states, actions, pmdc, alpha, zero_action)
        for t in range(1, length - 1):           # need s_t (info) and a future to predict
            kmax = min(alpha, length - 1 - t)
            if kmax < 1:
                continue
            absp_preds = absp_forward(states, actions, pmdc, t, kmax)
            for k in range(1, kmax + 1):
                true = states[t + k]
                m_absp = ee_metrics(absp_preds[k - 1], true)
                m_sbsp = ee_metrics(sbsp_snaps[t][k - 1], true)
                for metric in METRICS:
                    sink[("absp", metric)]["k"].append(k)
                    sink[("absp", metric)]["v"].append(m_absp[metric])
                    sink[("absp", metric)]["seed"].append(seed)
                    sink[("sbsp", metric)]["k"].append(k)
                    sink[("sbsp", metric)]["v"].append(m_sbsp[metric])
                    sink[("sbsp", metric)]["seed"].append(seed)
                n_anchor += 1

    env.close()
    track_cm = 100.0 * float(np.mean(track_errs)) if track_errs else float("nan")
    print(f"[seed {seed}] alpha={alpha} anchors={n_anchor} wm_loss={final_loss:.4f} "
          f"wm_pred_err(@alpha)={final_pred_err:.3f} true_tracking_err={track_cm:.1f}cm", flush=True)


def new_sink():
    return {(cond, metric): {"k": [], "v": [], "seed": []} for cond in CONDS for metric in METRICS}


def plot_primary(sink, alpha, delay_label, out_path, metric, seed_list):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    assert isinstance(ax, Axes)
    step_ms = 10
    ymax = 0.0
    for cond in CONDS:
        h, line, lo, hi = aggregate(sink, alpha, metric, cond)
        color = COND_COLORS[cond]
        ax.fill_between(h, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(h, line, color=color, lw=2.2, label=COND_LABELS[cond], zorder=3)
        for sd in seed_list:
            _, sl, _, _ = aggregate(sink, alpha, metric, cond, seeds=[sd])
            ax.plot(h, sl, color=color, lw=0.8, alpha=0.30, zorder=2)
        ymax = max(ymax, float(np.nanmax(hi)) if np.isfinite(np.nanmax(hi)) else 0.0)
    ax.axhline(EPS_CM, ls="--", color="0.35", lw=1.2, label=f"goal tolerance $\\epsilon$ = {EPS_CM:.0f} cm")
    ax.set_xlabel("delay horizon predicted, $k$ (env steps)")
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (cm)")
    ax.set_xlim(1, alpha)
    ax.set_ylim(0, ymax * 1.08 if ymax else None)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    n_seed = len(seed_list)
    ax.set_title(f"ABSP vs SBSP forward prediction error vs horizon\n"
                 f"FetchPush, {delay_label} ($\\alpha$={alpha}), IQM + IQR, "
                 f"{n_seed} seed{'s' if n_seed != 1 else ''}", fontsize=11)
    secax = ax.secondary_xaxis("top", functions=(lambda k: k * step_ms, lambda ms: ms / step_ms))
    secax.set_xlabel("delay horizon (ms)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def save_npz(sink, alpha, delay_label, seed_list, path):
    flat = {"__alpha": np.array([alpha]), "__delay": np.array([delay_label]),
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


def write_summary_csv(sink, alpha, out_path):
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "horizon_k", "cond", "iqm_cm", "iqr_lo_cm", "iqr_hi_cm"])
        for metric in METRICS:
            for cond in CONDS:
                h, line, lo, hi = aggregate(sink, alpha, metric, cond)
                for i, k in enumerate(h):
                    w.writerow([metric, int(k), cond, f"{line[i]:.4f}", f"{lo[i]:.4f}", f"{hi[i]:.4f}"])
    print(f"saved {out_path}")


def merge_npz(out_dir, tags):
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


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    p.add_argument("--delay-range", default="250-290")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--operator-model", default="operator_models/FetchPush-v2_SAC_seed0.zip")
    p.add_argument("--policy", default="pred_rl_models/PMDC/5/250-290ms/seed_0/"
                                       "24_step_SAC_FetchPush-RemotePDNorm-v0.zip")
    p.add_argument("--wm-warmup-steps", type=int, default=80000)
    p.add_argument("--wm-train-every", type=int, default=4)
    p.add_argument("--wm-polish-iters", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="diagnostics/absp_vs_sbsp")
    p.add_argument("--tag", default=None)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--replot", action="store_true")
    p.add_argument("--combine-tags", nargs="+", default=None)
    p.add_argument("--load-wm", action="store_true",
                   help="load the policy's persisted world model and skip warmup (on-distribution)")
    return p


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

    plot_primary(sink, alpha, delay_label, out_dir / f"{tag}_remote_ee.png", "remote_ee", seed_list)
    write_summary_csv(sink, alpha, out_dir / f"{tag}_summary.csv")


if __name__ == "__main__":
    main()
