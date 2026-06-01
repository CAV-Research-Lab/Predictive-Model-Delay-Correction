# CLAUDE.md — PMDC reproduction guide

This file is the operational guide for reproducing the **Predictive-Model Delay-Correction
(PMDC)** results from **thesis Chapter 3** ("Predictive Delay Correction for Stochastic
Teleoperation"). It captures exact commands **and** the pitfalls discovered while getting
the original `gym`-era code running on a modern `gymnasium` stack. Read the "Pitfalls"
section before changing anything — several traps here will silently produce wrong numbers.

> The parent repo guide (`/home/dook/PhD-thesis/CLAUDE.md`) still says this project installs
> via `requirements.txt` (pinned `gym==0.21`, `sb3==1.6`, `torch==1.11`). That pinned stack
> is **not** what runs here today. The code in this directory has been migrated to the
> versions listed below. Treat `requirements.txt` as historical.

---

## 1. What this project is

A local-remote teleoperation task: a **remote** robot arm must track a **local** operator's
end-effector over a network with communication delay. The remote agent (SAC) outputs PD-gains
`(Kp, Kd)`; the reward is the negative euclidean tracking error between the two end-effectors.
Base environment is a modified `FetchPush` (`FetchPush-RemotePDNorm-v0`, registered in
`robo_local_remote_env.py`, 50-step episodes, 22-dim obs, 2-dim action).

Three methods are compared:

| Method | State given to SAC | Dim @90-120ms / @250-290ms | Idea |
|--------|--------------------|----------------------------|------|
| **SAC**   | raw (stale) observation              | 22 / 22 | ignores delay → acts on old info |
| **A-SAC** | obs + **full** action history        | 44 / **78** | restores info, but state grows linearly with delay |
| **PMDC**  | **predicted future** obs + stochastic-range action history | 30 / **32** | world model absorbs the constant delay; only augment the stochastic part |

The thesis claim PMDC reproduces: **at short delay all three are comparable; as delay grows,
A-SAC's augmented state explodes and degrades, while PMDC stays small/relevant and keeps
tracking accurately and stably.**

---

## 2. Environment (what actually runs here)

```
python==3.10.12
gymnasium==0.28.1            # NOT classic `gym`
gymnasium-robotics==1.2.4    # provides FetchPush-v2
stable-baselines3==2.3.2
torch==2.5.1 (+cu124)        # CUDA strongly recommended
numpy==1.26.4
mujoco==2.3.7                # native bindings; mujoco_py is NOT installed
pandas==2.2.3                # plot_results.py
tensorboard==2.16.2          # plot_tensorboard.py (authoritative metric reader)
```

These are already installed in the user-site (`~/.local/lib/python3.10`). Do **not**
`pip install -r requirements.txt` — it would downgrade to the broken `gym 0.21` stack.

---

## 3. Exact reproduction (start to finish)

All commands run from this directory (`code/time-delay/Predictive-Model-Delay-Correction`).
Full run is ~3.5 h on an RTX 3080 (operator ~30 min, then 6 training runs).

### Step 0 — operator (local "expert") policy  [required, ~30 min on GPU]
The remote agent needs a local operator to track. The committed
`operator_models/FetchPush-v1_SAC_seed0.zip` **cannot be loaded** (see Pitfall A). Retrain:

```bash
python3 train_operator.py --steps 50000 --device cuda
# writes operator_models/FetchPush-v2_SAC_seed0.zip
```

`working_models.json` already points every env id at this checkpoint **without the `.zip`
suffix** (SB3 appends it — see Pitfall B):

```json
{ "FetchPush-v1": "operator_models/FetchPush-v2_SAC_seed0",
  "FetchPush-v2": "operator_models/FetchPush-v2_SAC_seed0" }
```

### Step 1 — train the three methods at both delays  [~3 h on GPU]
One process per (algorithm, delay). The long-delay (250-290) PMDC run is the slowest
(~15 fps; ensemble forward passes + recalibration per step). Run them in parallel if VRAM
allows (3×~1.2 GB), else sequentially:

```bash
OP=operator_models/FetchPush-v2_SAC_seed0.zip
for DELAY in 250-290 90-120; do
  for ALG in PMDC A-SAC SAC; do
    python3 delay_correcting_training.py \
      --delay-ranges $DELAY --algorithms $ALG \
      --steps 60000 --device cuda --operator-model $OP
  done
done
```

Outputs per run:
- `pred_rl_models/<ALG>/5/<DELAY>ms/seed_0/<act_delay>_step_SAC_*.zip` — **SAC policy only** (see Pitfall E)
- `pred_logs/<ALG>/5/<DELAY>ms/seed_0/SAC_*/` — TensorBoard logs (**authoritative metric**)
- `pred_results/<ALG>/5/<DELAY>ms/seed_0.csv` — sparse CSV (every 100 steps; coarser)

`--steps 80000` matches the thesis exactly; we used 60000 to save time and it already
reproduces the result. `--skip-existing` skips runs whose policy zip exists (handy for
resuming), but it will **not** re-run the metric fix if an old zip is present — delete stale
zips first if in doubt.

### Step 2 — read final numbers + summary figure  [authoritative]
```bash
# 1) dump final-window true-tracking stats from TensorBoard to /tmp/real_results.json
#    (this is the SAME metric the thesis Table 2 reports)
python3 - <<'PY'
from tensorboard.backend.event_processing import event_accumulator
import numpy as np, json
from pathlib import Path
out={}
for d in ['90-120ms','250-290ms']:
  out[d]={}
  for a in ['PMDC','A-SAC','SAC']:
    ld=str(sorted(Path(f'pred_logs/{a}/5/{d}/seed_0').iterdir())[-1])
    ea=event_accumulator.EventAccumulator(ld); ea.Reload()
    r=np.array([e.value for e in ea.Scalars('track/step_reward')])[-100:]
    out[d][a]=(float(np.mean(r)),float(np.std(r)))
json.dump(out,open('/tmp/real_results.json','w')); print(out)
PY

# 2) build the summary bar/line figure from that JSON (never hardcodes — see Pitfall F)
python3 final_summary.py        # -> PMDC_reproduction_summary.png

# 3) training curves (consistent metric, all three, full run)
python3 plot_tensorboard.py --delays 250-290ms 90-120ms --output PMDC_training_curves.png
```

### Optional — deterministic policy eval  (valid for SAC/A-SAC ONLY)
```bash
python3 evaluate_policies.py --delay 250-290 --n-episodes 20 \
  --operator-model operator_models/FetchPush-v2_SAC_seed0.zip
```
**PMDC's row will be garbage (~1.9 m).** This is expected — see Pitfall E. Trust the
TensorBoard `track/step_reward` numbers for PMDC, not this script.

---

## 4. Pitfalls (read before editing — each one silently breaks results)

**A. The committed operator checkpoint will not load.**
`operator_models/FetchPush-v1_SAC_seed0.zip` was pickled under classic `gym`; loading it
raises `ModuleNotFoundError: No module named 'gym'`. Do not try to shim `sys.modules['gym']`
— the observation space inside also fails. **Retrain with `train_operator.py`** (Step 0).

**B. `working_models.json` paths must omit the `.zip`.** SB3's `SAC.load` appends `.zip`.
A path ending in `.zip` becomes `...zip.zip` → `FileNotFoundError`.

**C. `FetchPush-v1` is unusable; use `v2`.** `v1` pulls in `mujoco_py` (not installed and not
worth installing). `robo_local_remote_env.py` makes `FetchPush-v2`; `working_models.json`
keeps a `FetchPush-v1` alias only so older references resolve.

**D. The gym→gymnasium API change is pervasive.** Already done in `wrappers_rd.py`,
`PMDC_wrapper.py`, `robo_local_remote_env.py`, `delay_correcting_training.py`,
`train_operator.py`, `delay_correcting_nn.py`. If you add/restore code, remember:
`reset()` returns `(obs, info)`; `step()` returns `(obs, reward, terminated, truncated, info)`
(5-tuple). The delay wrappers collapse `terminated/truncated` internally and re-expose a
5-tuple to SB3. `import gym` → `import gymnasium as gym`.

**E. PMDC's world model is NOT saved — deterministic re-eval of PMDC is invalid.**
The ensemble dynamics model lives inside the `PMDC` wrapper and is trained **online**; the
saved `.zip` contains only the SAC policy (`policy.pth` + optimizers). Reloading a PMDC
policy into a fresh wrapper gives it a **random** world model, so it receives garbage
future-state predictions and flails (~1.9 m). Therefore:
- The **valid** PMDC performance number is the **training-time** true-tracking reward
  (`track/step_reward` in TensorBoard = `REnvPDNormObs.reward`, the real euclidean error).
  This is exactly what thesis Table 2 reports ("mean over the final training episode").
- `evaluate_policies.py` is only meaningful for SAC and A-SAC (no world model).
- If you want a valid offline PMDC eval, you must persist the ensemble: call
  `PMDC.save_models()` after training and reload it in the eval wrapper (`pretrain=True`
  loads `./models/<env_id>/2-128-1_step_prediction_sd_*.pt`). Not wired up by default.

**F. Never hardcode results into a plotting script.** An earlier version of
`final_summary.py` had numbers typed in from memory that disagreed with the actual eval by
30×. `final_summary.py` now reads `/tmp/real_results.json`. Keep it that way — plots must be
derived from logged data only.

**G. The original CSV logging used inconsistent metrics across algorithms.** The committed
`pred_results/seed0_summary.csv` is misleading: PMDC's column logged its internal
predicted-state / delayed reward while SAC/A-SAC logged the live reward — not comparable, and
it makes PMDC look *worse*. The callback in `delay_correcting_training.py` was fixed to log
`REnvPDNormObs.reward` (the true tracking error at the base env) for **all** algorithms, under
`track/step_reward` / `track/episode_reward`. Use TensorBoard, not the old summary CSV.

**H. `learning_starts=20000`.** The first 20k steps are random exploration (flat curves);
SAC only starts learning after. Don't interpret the first 20k as failure.

**I. PMDC converges late then sharply.** Expect PMDC to *lag* A-SAC for the first ~25-45k
steps while the world model calibrates, then drop sharply to near-zero error. At 250-290ms
the crossover/breakthrough was around step ~45k in our run. The critic loss is the tell:
PMDC's critic converges to <0.1 while A-SAC/SAC stay >2 at long delay (smaller, relevant
state → stable Q-function).

**J. Use the GPU.** `--device cuda`. CPU operator training is ~5× slower. PMDC at 250-290ms
runs ~15 fps even on GPU (ensemble + recalibration); SAC/A-SAC ~18-90 fps.

**K. Output paths are keyed by `algorithm/delay/seed` but NOT by env — different
environments collide.** `pred_logs/` and `pred_results/` use
`<algorithm>/<n_models>/<delay>/seed_<seed>/`, with **no env id**. So running a second
environment at the same delay writes into the same tree: TensorBoard quietly adds another
`SAC_<n>` subdir (confusing), and **the CSV at `pred_results/.../seed_<seed>.csv` is
overwritten**. (Policy `.zip`s are safe — their filename embeds the env id.) **Always pass
`--output-dir runs_<EnvName>` for any non-FetchPush environment** to isolate everything under
that prefix. FetchPush is the default project at root (`--output-dir .`); each other env gets
its own `runs_<EnvName>/`. To tell colliding TB runs apart after the fact, match the PID in
the event filename (`events.out.tfevents.<unixtime>.<host>.<PID>.0`) to the launch log.

---

## 5. Results we obtained (seed 0, 60k steps)

Metric = final-episode true tracking error (`track/step_reward`, last 100 logged episodes),
mean ± std, in cm. Thesis = Table 2 reference.

| Delay | PMDC (ours) | A-SAC | SAC | Thesis PMDC / A-SAC / SAC |
|-------|-------------|-------|-----|---------------------------|
| 90-120ms  | **4.5 ± 1.6**  | 5.6 ± 5.4  | 6.1 ± 17.6 | 3.0 / 3.4 / 5.3 |
| 250-290ms | **5.3 ± 5.3**  | 25.7 ± 25.4 | 21.8 ± 31.8 | 4.3 / 15 / 25 |

Takeaway (matches thesis): short delay → all comparable; long delay → **PMDC ~4-5× lower
error and far lower variance**. Mechanism confirmed via state dims (32 vs 78) and critic-loss
convergence (PMDC <0.1 vs A-SAC ~2.2 at 250-290ms).

**Caveats on our numbers:** single seed; 60k (not 80k) steps; our operator only reaches
~7% FetchPush success (the thesis used a stronger expert), which inflates variance and is why
our SAC ≈ A-SAC at long delay rather than SAC being clearly worst. The PMDC-vs-rest gap is
robust to all of this.

---

## 6. File map

Source (migrated this session; tracked by git):
- `wrappers_rd.py` — random action/observation delay wrappers (Bouteiller et al. `rlrd`-style).
  `UnseenRandomDelayWrapper` (raw, for SAC), `AugmentedRandomDelayWrapper` (obs+action history).
- `PMDC_wrapper.py` — the PMDC `gym.Wrapper`: SBSP future-state buffer, recalibration, online
  ensemble training (`learn`), reward from predicted state. **World model not persisted by default.**
- `delay_correcting_nn.py` — `DCNN` (2×128 ReLU MLP, Huber loss) ensemble member; `run()` pretrain helper.
- `robo_local_remote_env.py` — `REnvPDNormObs` local-remote env + `FetchPush-RemotePDNorm-v0` registration.
- `delay_correcting_training.py` — entry point. Delay parsing, env builder per algorithm, SAC training,
  **fixed metric callback** (`track/step_reward`).
- `train_operator.py` — trains the local operator SAC (HER) on `FetchPush-v2`.

Tooling (created this session; untracked):
- `plot_tensorboard.py` — **authoritative** plotter; reads `track/step_reward` from TensorBoard.
- `final_summary.py` — summary bar/line figure; reads `/tmp/real_results.json` (no hardcoding).
- `evaluate_policies.py` — deterministic rollout eval. **Valid for SAC/A-SAC only** (Pitfall E).
- `plot_results.py` — CSV-based plotter; coarser (100-step sampling), kept for convenience.

Data / artifacts:
- `pred_logs/` — TensorBoard (authoritative). `pred_results/` — sparse CSVs. `pred_rl_models/` — policy zips.
- `operator_models/FetchPush-v2_SAC_seed0.zip` — the operator. `working_models.json` — env→operator map.
- `pred_results_old/` — backup of the original gym-era (inconsistent-metric) results, for reference only.
- Keep only these plots: `PMDC_reproduction_summary.png`, `PMDC_training_curves.png`.

## 7. Delay-config decoder (`delay_correcting_training.py`)

`"250-290"` → `min_ms=250, max_ms=290`:
- `act_delay = min_ms/10 - 1 = 24` constant action-delay steps (the part PMDC's model predicts).
- `obs_delay_range = range(0, (max-min)/10 + 1) = range(0,5)` stochastic observation delay (the part
  PMDC augments and A-SAC/SAC must also absorb).

`"90-120"` → `act_delay=8`, `obs_delay_range=range(0,4)`. 10 ms per env step. So "250-290ms" =
240 ms constant + up to 40 ms stochastic, matching the thesis's "240ms + 10-50ms" setup.

---

## 8. Running a different environment (FetchSlide / FetchPickAndPlace)

Adding another environment is genuinely cheap because the only env-specific code,
`format_obs()` in `robo_local_remote_env.py`, slices fixed indices out of the **25-dim obs /
4-dim action** layout that `FetchPush`, `FetchSlide`, and `FetchPickAndPlace` all share. The
PD controller (`np.append(pd_value, [0])`) and the reward are layout-agnostic too. **FetchReach
will NOT work** (10-dim obs, no object) without rewriting `format_obs`.

Already wired up: `register_local_remote_envs()` registers `FetchSlide-RemotePDNorm-v0` and
`FetchPickAndPlace-RemotePDNorm-v0` (dict `LOCAL_REMOTE_ENVS`), and `working_models.json` maps
their base envs to operator checkpoints. To add yet another same-layout env, add one line to
each.

Recipe (FetchPickAndPlace shown; swap the name for FetchSlide):

```bash
# 0) train an operator for that base env (~30 min GPU). Tracking generalises even with a weak
#    operator (~6% task success) because the remote agent only follows the gripper position,
#    not the task — see the thesis's "generalises to other tasks" argument.
python3 train_operator.py --env-id FetchPickAndPlace-v2 --steps 50000 --device cuda \
  --output operator_models/FetchPickAndPlace-v2_SAC_seed0.zip

# 1) sanity-check the chain BEFORE long training (seconds): the existing FetchPush operator
#    loads into the new env (identical spaces) just to validate format_obs + PD + PMDC buffer.
#    (build_training_env for SAC/A-SAC/PMDC; assert finite reward, correct obs dims.)

# 2) train — ALWAYS pass --output-dir for non-FetchPush envs (Pitfall K) to avoid collisions.
OP=operator_models/FetchPickAndPlace-v2_SAC_seed0.zip
for ALG in PMDC A-SAC SAC; do
  python3 delay_correcting_training.py \
    --env-id FetchPickAndPlace-RemotePDNorm-v0 \
    --delay-ranges 250-290 --algorithms $ALG --steps 60000 --device cuda \
    --operator-model "$OP" --output-dir runs_FetchPickAndPlace
done

# 3) read/plot from the isolated tree
python3 plot_tensorboard.py --log-base runs_FetchPickAndPlace/pred_logs \
  --delays 250-290ms --output FetchPickAndPlace_250-290ms.png
```

**Results we recorded on FetchPickAndPlace (seed 0):**
- 90-120ms, 30k steps (smoke): clean run. A-SAC converged to 11.4 cm; SAC 48 cm, PMDC 52 cm
  still pre-breakthrough at 30k (PMDC critic ~0.05 — the Pitfall I signature, so short delay
  was on track).
- **250-290ms, 60k steps: PMDC did NOT reproduce its FetchPush advantage. It diverged.**
  Final tracking: **A-SAC 22.5 cm, SAC 39.8 cm, PMDC 100.7 cm** (PMDC got *worse* over the run:
  51 cm at 30k → 100 cm at 55k). PMDC's prediction critic was ~4.0 (vs ~0.05 on FetchPush).

**Why PMDC fails here, and the key lesson (Pitfall L below).** PMDC replaces A-SAC's huge
augmented state with a *predicted* future state, so it trades the curse of dimensionality for a
hard dependency on **world-model accuracy**. FetchPickAndPlace with our weak 50k-step operator
produces an erratic, gripper-driven 3D reference; predicting it **24 steps ahead** (the
250-290ms horizon) compounds error badly, so the policy acts on future states that don't match
reality and destabilises. The model-free baselines (A-SAC/SAC) use *real* delayed states, so
they degrade gracefully instead. Tell-tale: at the 8-step horizon (90-120ms) PMDC's critic
converged, but at the 24-step horizon it blew up — a long-horizon-prediction failure, not a bug.

**To give PMDC a fair shot on FetchPickAndPlace:** train a strong operator first (HER, ~10⁶
steps, until success-rate is high) so the reference trajectory is smooth and learnable, then
re-run 250-290ms. Do NOT assume PMDC's FetchPush win transfers to every task — it requires a
predictable/learnable reference. This is a genuine result, not a misconfiguration.

**Pitfall L. PMDC's advantage is conditional on a learnable world model.** Its critic-loss
converging means its Q-function is consistent *in the predicted state space* — that says
nothing about whether the predictions match reality. On hard-to-model dynamics or with an
erratic operator over a long horizon, PMDC can underperform plain SAC. Always sanity-check the
actual `track/step_reward`, not just the critic loss.
