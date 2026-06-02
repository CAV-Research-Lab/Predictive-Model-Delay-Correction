"""
Train the local "operator" (expert) policy that the remote agent tracks.

Optimised pipeline for reaching high success on hard Fetch tasks:
  - Train on the *Dense* reward variant (continuous -distance signal) + HER.
    Dense was decisive on FetchSlide (sparse caps ~4%; dense+HER reaches 20-80%).
  - gamma 0.95 (standard for 50-step Fetch episodes).
  - EvalCallback on the SPARSE counterpart env so "best mean reward" == "best success",
    and the best checkpoint is saved + copied to --output (so --output is the best operator).
  - CheckpointCallback for crash safety on long runs.

The operator is used in REnvPDNormObs (which runs the sparse FetchX-v2 env); dense vs sparse
share identical dynamics/obs/action, so a dense-trained operator runs there unchanged.
"""
import argparse
import random
import shutil
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 - registers the Fetch envs
import numpy as np
import torch
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser(description="Train the local operator policy (SAC+HER, dense-reward).")
    p.add_argument("--env-id", default="FetchPushDense-v2",
                   help="Train on the Dense variant; eval is auto-run on the sparse counterpart.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--no-her", action="store_true", help="disable HER (HER is kept on by default)")
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--ent-coef", default="auto",
                   help='"auto" (default), "auto_<init>", or a fixed float; raise to fight entropy collapse')
    p.add_argument("--target-entropy", default="auto",
                   help='"auto" (=-dim(A)) or a float; raise above -dim(A) to keep ent_coef from collapsing')
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--output", default="operator_models/FetchPush-v2_SAC_seed0.zip")
    p.add_argument("--init-from", default=None,
                   help="warm-start the policy from a saved .zip (e.g. a best_model after a crash/restart)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="pmdc-fetch-operators")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    seed_everything(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out.parent / (out.stem + "_run")
    best_dir = run_dir / "best"
    ckpt_dir = run_dir / "checkpoints"
    for d in (best_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"{args.env_id}-seed{args.seed}", config=vars(args),
            sync_tensorboard=True, mode=args.wandb_mode, dir=str(run_dir),
            reinit=True,
        )
        print(f"W&B: {wandb_run.url if wandb_run else '(disabled)'}")

    # Eval on the SPARSE counterpart so best-mean-reward tracks success, not distance.
    eval_env_id = args.env_id.replace("Dense", "")
    train_env = Monitor(gym.make(args.env_id))
    eval_env = Monitor(gym.make(eval_env_id))
    train_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 10_000)

    her_kwargs = {} if args.no_her else {
        "replay_buffer_class": HerReplayBuffer,
        "replay_buffer_kwargs": {"n_sampled_goal": 4, "goal_selection_strategy": "future"},
    }
    policy_kwargs = {"net_arch": args.net_arch}
    if args.optimizer == "adamw":
        policy_kwargs["optimizer_class"] = torch.optim.AdamW
        policy_kwargs["optimizer_kwargs"] = {"weight_decay": args.weight_decay}
    elif args.weight_decay > 0:
        policy_kwargs["optimizer_kwargs"] = {"weight_decay": args.weight_decay}
    ent_coef = args.ent_coef if str(args.ent_coef).startswith("auto") else float(args.ent_coef)
    target_entropy = "auto" if str(args.target_entropy) == "auto" else float(args.target_entropy)

    if args.init_from and Path(args.init_from).exists():
        # Warm-start after a crash: policy weights are preserved; the replay buffer restarts.
        model = SAC.load(args.init_from, env=train_env, device=args.device)
        print(f"Warm-started policy from {args.init_from}")
    else:
        model = SAC(
            "MultiInputPolicy",
            train_env,
            **her_kwargs,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            learning_starts=args.learning_starts,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tb"),
            device=args.device,
            verbose=1,
        )
    callbacks = [
        CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(ckpt_dir), name_prefix="operator"),
        EvalCallback(eval_env, best_model_save_path=str(best_dir), log_path=str(best_dir),
                     eval_freq=args.eval_freq, n_eval_episodes=args.eval_episodes,
                     deterministic=True, render=False),
    ]
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(verbose=0))
    print(f"Training {args.env_id} (eval on {eval_env_id}) | steps={args.steps} | net={args.net_arch} | "
          f"gamma={args.gamma} HER={not args.no_her} batch={args.batch_size}")
    model.learn(total_timesteps=args.steps, log_interval=10, callback=callbacks)

    model.save(str(out))  # final
    best = best_dir / "best_model.zip"
    if best.exists():
        shutil.copy(best, out)  # prefer the best-by-success checkpoint as the operator
        print(f"Copied best-by-success model -> {out}")
    train_env.close(); eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done. Operator: {out}  | best/checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
