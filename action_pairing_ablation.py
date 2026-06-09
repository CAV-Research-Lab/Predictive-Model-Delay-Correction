#!/usr/bin/env python3
"""Ablation: is the world model's one-step error floor caused by the action-delay mismatch?

PMDC trains its world model on pairs (s_{t-1}, a_t) -> s_t, but the transition is actually
driven by the action applied ~alpha steps ago (the inner UnseenRandomDelayWrapper holds each
command for the action delay). So the model is fed an action that did *not* cause the
transition. This script tests whether that mismatch is what limits the one-step prediction
floor (the ~5 cm seen in openloop_rollout_error.py).

It collects on-distribution transitions, then trains three otherwise-identical world models
that differ ONLY in the action fed:
  * current  : (s_{t-1}, a_t)          -- PMDC's actual (decorrelated) pairing
  * delayed  : (s_{t-1}, a_{t-alpha})  -- the action actually applied (correct causal input)
  * none     : (s_{t-1}, 0)            -- state only (action zeroed)
and compares their one-step remote-EE prediction error (cm) on held-out episodes.

Reading: if delayed << current ~ none, the action-delay mismatch dominates the floor (and a
WM given the correct input would predict far better, consistent with a deterministic system).
If all three are similar, the floor is from partial observability / model capacity instead.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PMDC_FIX_PREVOBS", "1")

import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

import robo_local_remote_env  # noqa: F401,E402
from delay_correcting_nn import DCNN  # noqa: E402
from delay_correcting_training import build_training_env, parse_delay_range, seed_everything  # noqa: E402
from openloop_rollout_error import REMOTE_EE, choose_action, find_pmdc  # noqa: E402

COND_LABELS = {"current": "current $a_t$ (PMDC pairing)",
               "delayed": "applied $a_{t-\\alpha}$ (correct)",
               "none": "no action (state only)"}
COND_COLORS = {"current": "#d1495b", "delayed": "#1f77b4", "none": "#999999"}


def collect_episodes(env, pmdc, policy, n_steps, train_every):
    """Run the policy + online WM (so trajectories are on-distribution) and record per-episode
    sequences of (true state, action reaching PMDC)."""
    pmdc.train_every = train_every
    captured = {"a": np.zeros_like(np.asarray(pmdc.action_space.sample(), dtype=np.float32))}
    inner_step = pmdc.step

    def recording_step(action):
        captured["a"] = np.asarray(action, dtype=np.float32).copy()
        return inner_step(action)

    pmdc.step = recording_step

    episodes, ep = [], []
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs, _, term, trunc, _ = env.step(choose_action(policy, obs, env))
        ep.append((np.asarray(pmdc.prev_obs, dtype=np.float32).copy(), captured["a"].copy()))
        if term or trunc:
            episodes.append(ep)
            ep = []
            obs, _ = env.reset()
    if ep:
        episodes.append(ep)
    pmdc.train_every = 0
    pmdc.freeze_wm = 1
    return episodes


def make_tuples(episodes, alpha):
    """(s_{t-1}, a_t, a_{t-alpha}, s_t) for each within-episode transition with a valid
    applied (delayed) action."""
    sp, ac, ad, sn = [], [], [], []
    for ep in episodes:
        s = [x[0] for x in ep]
        a = [x[1] for x in ep]
        for k in range(alpha, len(s)):       # predict s[k] from s[k-1]
            sp.append(s[k - 1])
            ac.append(a[k])                  # current action (PMDC pairing)
            ad.append(a[k - alpha])          # action applied alpha steps later == drives s[k]
            sn.append(s[k])
    f = lambda L: np.asarray(L, dtype=np.float32)
    return f(sp), f(ac), f(ad), f(sn)


def train_wm(s_in, a_in, s_out, device, n_iter, batch, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DCNN(beta=5e-5, input_dims=s_in.shape[1], n_actions=a_in.shape[1],
                 layer_size=128, n_layers=2)
    model.device = torch.device(device)
    model.to(model.device)
    X = np.concatenate([s_in, a_in], axis=1)
    n = len(X)
    model.train()
    for _ in range(n_iter):
        idx = np.random.choice(n, batch)
        model.learn(X[idx], s_out[idx])
    model.eval()
    return model


def eval_onestep_remote_ee(model, s_in, a_in, s_out):
    X = np.concatenate([s_in, a_in], axis=1)
    with torch.no_grad():
        pred = model.forward(torch.tensor(X, device=model.device, dtype=torch.float32)).cpu().numpy()
    return np.linalg.norm(pred[:, REMOTE_EE] - s_out[:, REMOTE_EE], axis=1) * 100.0  # cm


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(np.floor(n * 0.25))
    return x[k:n - k].mean() if n - 2 * k > 0 else x.mean()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    p.add_argument("--delay-range", default="250-290")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--operator-model", default="operator_models/FetchPush-v2_SAC_seed0.zip")
    p.add_argument("--policy", default="pred_rl_models/PMDC/5/250-290ms/seed_0/"
                                       "24_step_SAC_FetchPush-RemotePDNorm-v0.zip")
    p.add_argument("--collect-steps", type=int, default=30000)
    p.add_argument("--collect-train-every", type=int, default=2)
    p.add_argument("--n-iter", type=int, default=6000, help="grad steps per ablation WM")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--net-seeds", type=int, default=3, help="WM inits averaged per condition")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="diagnostics/action_ablation")
    p.add_argument("--tag", default=None)
    return p


def main():
    args = build_parser().parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    delay_config = parse_delay_range(args.delay_range)
    alpha = delay_config.act_delay
    seed_everything(args.seed)

    env = build_training_env("PMDC", args.env_id, delay_config, args.seed,
                             n_models=5, pretrain=False, operator_model=args.operator_model)
    pmdc = find_pmdc(env)
    pmdc.fix_prev_obs = True
    policy = SAC.load(args.policy, device=args.device) if args.policy else None

    print(f"[collect] {args.collect_steps} steps (alpha={alpha}) ...", flush=True)
    episodes = collect_episodes(env, pmdc, policy, args.collect_steps, args.collect_train_every)
    env.close()

    on_dist = episodes[len(episodes) // 2:]          # second half == on-distribution
    n_test = max(1, len(on_dist) // 5)
    train_eps, test_eps = on_dist[:-n_test], on_dist[-n_test:]
    sp_tr, ac_tr, ad_tr, sn_tr = make_tuples(train_eps, alpha)
    sp_te, ac_te, ad_te, sn_te = make_tuples(test_eps, alpha)
    print(f"[data] episodes total={len(episodes)} on-dist={len(on_dist)} "
          f"train_tuples={len(sp_tr)} test_tuples={len(sp_te)}", flush=True)

    conds = {
        "current": (ac_tr, ac_te),
        "delayed": (ad_tr, ad_te),
        "none": (np.zeros_like(ac_tr), np.zeros_like(ac_te)),
    }
    results = {}
    for cond, (a_tr, a_te) in conds.items():
        errs = []
        for ns in range(args.net_seeds):
            model = train_wm(sp_tr, a_tr, sn_tr, args.device, args.n_iter, args.batch,
                             seed=1000 * args.seed + ns)
            errs.append(eval_onestep_remote_ee(model, sp_te, a_te, sn_te))
        errs = np.concatenate(errs)
        results[cond] = errs
        print(f"[{cond:8s}] one-step remote-EE error (cm):  "
              f"IQM={iqm(errs):6.2f}  median={np.median(errs):6.2f}  mean={errs.mean():6.2f}",
              flush=True)

    # bar chart
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.env_id.split('-')[0]}_{delay_config.label}_seed{args.seed}"
    try:
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=160)
    order = ["current", "delayed", "none"]
    xs = np.arange(len(order))
    vals = [iqm(results[c]) for c in order]
    los = [np.percentile(results[c], 25) for c in order]
    his = [np.percentile(results[c], 75) for c in order]
    yerr = np.array([[v - lo for v, lo in zip(vals, los)], [hi - v for v, hi in zip(vals, his)]])
    ax.bar(xs, vals, color=[COND_COLORS[c] for c in order], alpha=0.85,
           yerr=yerr, capsize=5, edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v, f" {v:.1f}", va="bottom", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS[c] for c in order], fontsize=9)
    ax.set_ylabel("one-step remote-EE prediction error (cm)")
    ax.set_title(f"Action-pairing ablation -- FetchPush {delay_config.label} ($\\alpha$={alpha})\n"
                 f"one-step world-model error (IQM + IQR), held-out episodes", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}.png", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out_dir / f"{tag}.npz", **{c: results[c] for c in order},
                        alpha=np.array([alpha]))
    print(f"saved {out_dir / f'{tag}.png'}")


if __name__ == "__main__":
    main()
