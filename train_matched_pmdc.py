#!/usr/bin/env python3
"""Train a PMDC policy and PERSIST its world model (matched pair), fixing Pitfall E.

PMDC's ensemble world model is normally trained online and discarded, so a saved policy can
only ever be paired with a *fresh* WM -> off-distribution diagnostics. This trains policy+WM
together and saves BOTH: the SAC .zip and the ensemble .pt files under
./models/<env_id>/2-128-1_step_prediction_sd_<i>.pt, which PMDC(pretrain=True) reloads. The
diagnostics can then load the matched pair and run on-distribution.

Uses the stability-investigation config that converges (PMDC_FIX_PREVOBS=1, gamma=0.95,
target_entropy=-1) so the matched pair is a *good* policy, not a failed seed.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("PMDC_FIX_PREVOBS", "1")

import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

import robo_local_remote_env  # noqa: F401,E402
from delay_correcting_training import (  # noqa: E402
    TensorboardCallback, build_training_env, parse_delay_range, seed_everything,
)
from openloop_rollout_error import find_pmdc  # noqa: E402


def save_world_model(pmdc, env_id):
    """Persist the ensemble to the paths PMDC(pretrain=True) reads."""
    out = Path("models") / env_id
    out.mkdir(parents=True, exist_ok=True)
    n_layers, layer_size = pmdc.n_layers, pmdc.layer_size
    for i, model in enumerate(pmdc.dc_models):
        torch.save(model.state_dict(), out / f"{n_layers}-{layer_size}-1_step_prediction_sd_{i}.pt")
    # params pickle so a reload reconstructs the same architecture
    params = {"beta": 5e-5, "input_dims": pmdc.observation_space.shape[0],
              "n_actions": pmdc.action_space.shape[0], "layer_size": layer_size, "n_layers": n_layers}
    pickle.dump(params, open(out / "1_step_params.pickle", "wb"))
    print(f"saved world model ({pmdc.n_models} members) to {out}/")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="FetchPush-RemotePDNorm-v0")
    p.add_argument("--delay-range", default="250-290")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--operator-model", default="operator_models/FetchPush-v2_SAC_seed0.zip")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--target-entropy", type=float, default=-1.0)
    p.add_argument("--output-dir", default="matched_pmdc")
    args = p.parse_args()

    delay_config = parse_delay_range(args.delay_range)
    seed_everything(args.seed)
    env = build_training_env("PMDC", args.env_id, delay_config, args.seed,
                             n_models=args.n_models, pretrain=False, operator_model=args.operator_model)
    pmdc = find_pmdc(env)
    pmdc.fix_prev_obs = True

    out_dir = Path(args.output_dir)
    tb = out_dir / "tb" / f"seed_{args.seed}"
    model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=str(tb),
                buffer_size=20000, learning_rate=3e-4, gamma=args.gamma,
                target_entropy=args.target_entropy, device=args.device, learning_starts=20000)
    print(f"[train] PMDC {args.env_id} {delay_config.label} seed={args.seed} steps={args.steps} "
          f"gamma={args.gamma} tent={args.target_entropy}", flush=True)
    model.learn(total_timesteps=args.steps, log_interval=1, callback=TensorboardCallback(env=env))

    policy_path = out_dir / f"policy_{delay_config.label}_seed{args.seed}.zip"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(policy_path))
    save_world_model(pmdc, args.env_id)
    print(f"saved policy to {policy_path}", flush=True)
    print(f"[done] final track/step_reward={getattr(pmdc.env.env, 'reward', float('nan'))}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
