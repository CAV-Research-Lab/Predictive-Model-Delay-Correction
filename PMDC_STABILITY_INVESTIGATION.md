# PMDC training-stability investigation

**Author:** Claude (autonomous session)  **Started:** 2026-06-03 ~22:30  **Env:** FetchSlide-RemotePDNorm-v0, delay 250-290 ms, single operator (67%).

This document logs the investigation into PMDC's seed-dependent training instability, the
world-model diagnostics added, experiments run, and recommendations. Newest results at the
bottom of the experiment log.

---

## TL;DR (for review)

- **Your question - is the world-model learning rate the issue? No.** Instrumenting the ensemble
  (new `wm/pred_err`, `wm/ensemble_loss`) shows it calibrates *identically well* in converged and
  failed seeds (pred_err 2.5 -> 0.4, loss -> 0.017). The world model is not the bottleneck.
- **Real bug found:** `prev_obs` was frozen per episode, so the 1-step dynamics model trained on
  stale (state, action, next_state) pairs. Fixed (`PMDC_FIX_PREVOBS`). Affects the published method.
- **The instability is SAC** getting stuck in *stable poor local optima* on the predicted-state
  MDP - not the world model, and not critic divergence per se (that's only a high-LR symptom).
- **Best config found:** prev_obs fix + `gamma=0.95` (cleans up the critic, ~10x lower critic loss)
  + `target_entropy=-1` (moderate extra exploration) -> **3/5 seeds converge** (up from 1-2/5) with
  clean training. Too much entropy (tent=0) hurts; lower LR under-trains; longer training under test.
- **Bottom line:** improved and de-risked, but PMDC on this task remains somewhat seed-variable -
  report multiple seeds. All changes are env-var-gated and non-breaking (defaults = original code).

---

## 1. Problem statement

On FetchSlide at 250-290 ms, PMDC's tracking error is excellent **when it converges** (~5 cm,
~5x better than A-SAC) but **only converges on a minority of seeds**. Baseline (original code),
verified by re-running solo (no contention):

| method @250-290 | per-seed err (cm) | converged |
|---|---|---|
| **PMDC** | 5.0, 4.7, 43.8, 47.7, 54.1 | **2/5** |
| A-SAC | 20.0, 20.7, 19.5, 36.0, 27.7 | 5/5 |
| SAC | 32.6, 47.3, 30.7, 33.8, 26.5 | 5/5 |

The instability is **intrinsic, not a parallelism artifact**: the 3 failed seeds were re-run
solo and all failed again (s2=99.9, s3=99.7, s4=38.4 cm — failed seeds often diverge *worse*
solo). A-SAC/SAC are reliable; only PMDC is fragile.

Failure signature: SAC critic loss stays pinned (~1.0 at short delay) and tracking error never
improves / drifts up. Hypothesis: PMDC trains SAC on a reward derived from an **online,
non-stationary, sometimes-saturated** world-model prediction; on unlucky seeds the coupled
world-model/policy loop never escapes.

---

## 2. Findings so far

### F1 — `prev_obs` data bug in world-model training (confirmed, partial fix)
In `PMDC.step()`, `self.prev_obs` was **never updated** — it was set once per episode and frozen.
So the 1-step dynamics model was trained on `(stale_state, action_t) -> real_state_{t+1}` pairs:
the input state did not correspond to the action or target. The model could only learn
`action -> average-next-state`, not real dynamics.

- Fix (`PMDC_FIX_PREVOBS=1`): update `prev_obs = observation` each step.
- Effect on the 3 failed seeds (solo): **1/3 rescued** (s2: 99.9 -> 5.9 cm; s3, s4 still fail).
- Verdict: a **real bug and a real partial improvement**, but not sufficient alone.

### F2 — More-frequent world-model training is counterproductive (confirmed)
`PMDC_TRAIN_EVERY=1` (train the ensemble every env step instead of once per episode), combined
with the prev_obs fix, made a **previously-converged** seed **regress** (s2: 5.9 -> 37.0 cm).
Reason: training the model every step means the **reward function shifts every step** -> *more*
non-stationarity for SAC, not less. The once-per-episode cadence keeps the reward stationary
*within* an episode. So "train the model more" is the wrong lever; the right target is a reward
that is accurate **and stationary**.

### F3 — Instrumentation added (this session)
The world model was previously a black box (only downstream tracking error was logged). Added
two TensorBoard signals (env-var gated knobs, all default to original behaviour):
- `wm/ensemble_loss` — mean ensemble Huber training loss.
- `wm/pred_err` — delay-horizon prediction error `||actual_obs - predicted_obs||` each step.
- `PMDC_WM_LR` — override the ensemble Adam learning rate (default `beta=5e-5`).

Knobs live in `PMDC_wrapper.py` (`PMDC_FIX_PREVOBS`, `PMDC_TRAIN_EVERY`, `PMDC_WM_LR`,
`PMDC_FREEZE_WM`); logging in `delay_correcting_training.py` callback. All non-breaking.

### F4 — World-model LR is NOT the bottleneck; instability is SAC-side (KEY FINDING)
With instrumentation (experiment E0: 5 seeds, prev_obs fix, baseline LR), the world model
calibrates **identically well in converged and failed seeds**: `wm/pred_err` drops 2.5 -> 0.4-0.5
and `wm/ensemble_loss` collapses 0.29 -> ~0.017 in *every* seed, regardless of outcome (1/5
converged). So the ensemble is already well-trained even when the run fails — **raising the
world-model learning rate would not help.** The instability lives in **SAC**: given an accurate
reward it still converges only stochastically (~1/5), whereas A-SAC/SAC baselines converge 5/5 on
the same task. The differentiator is the SAC random seed, not the world model.

Leading remaining suspect: **reward non-stationarity** — the reward shifts as the model calibrates
(2.5 -> 0.4) *during* SAC learning (consistent with F2, where per-step updates made it worse).

| E0 seed | tracking err | wm/pred_err | wm/loss | outcome |
|---|---|---|---|---|
| 0 | 5.6 | 2.6->0.4 | 0.285->0.016 | converged |
| 1 | 83.8 | 2.5->0.4 | 0.251->0.019 | failed |
| 2 | 48.4 | 2.3->0.4 | 0.296->0.016 | failed |
| 3 | 89.8 | 2.3->0.5 | 0.292->0.021 | failed |
| 4 | 42.2 | 2.3->0.5 | 0.303->0.018 | failed |

### F5 — Failure signature is SAC critic divergence (KEY)
SAC-side metrics (E1) cleanly separate converged from failed seeds by the **critic loss**:
converged (s0,s1) keep `critic_loss` bounded ~0.9-1.25; failed (s2,s3,s4) let it **grow to
~1.9-2.45** (value overestimation / critic divergence). `ent_coef` collapses identically in all
seeds (~0.02), so entropy collapse is NOT the differentiator. => the lever is SAC critic
stability: lower learning rate, larger replay buffer, or lower gamma (remote SAC uses 0.99).

### F6 — Failed seeds are stuck in STABLE poor local optima (not divergence)
Stabilising the critic does NOT fix convergence: E2 (lr->1e-4) bounded the critic (~0.6) but gave
**0/5** (under-trained); E3 (gamma->0.95) collapsed critic loss to **~0.13 in ALL seeds** yet still
**2/5**, with failed seeds sitting at 40-89 cm on a LOW, converged critic loss = a *stable poor
optimum*, not divergence. So F5's "critic divergence" is a high-LR symptom; the deeper issue is SAC
getting stuck in a poor optimum on the predicted MDP and only escaping **stochastically** (the late
"breakthrough" - converged seeds' real error drops only in the final ~10k steps). This is an
**exploration / optimisation-landscape problem**. Model-exploitation is ruled out (the predicted
reward does not climb while real tracking stays bad; both plateau). Testing raised entropy (E4).

### F7 — The reward is OPTIMISTIC for failed policies (likely the true cause)
Mining ALL logged metrics on E4 (converged s0,s1,s2 vs failed s3,s4): the world model is again NOT
the differentiator (`wm/pred_err` separation 0.07). But the PREDICTED reward SAC optimises vs the
REAL tracking diverges sharply:
- **failed** seeds: predicted episode reward -9.9 (~0.20 m/step) but REAL is 0.55 m/step -> the
  model tells the agent it is doing **~3x better than reality**.
- **converged** seeds: predicted -5.4 (0.11/step) vs real 0.067/step -> mildly *pessimistic*.

So when SAC drifts into a poor policy, the world model **under-predicts the tracking error -> no
gradient to improve -> stuck** (the F6 "local optimum"). This is **model-optimism / exploitation**,
a classic model-based-RL failure. Candidate fix: an **ensemble-disagreement pessimism penalty**
(`PMDC_PESSIMISM`) - penalise the reward where the ensemble is uncertain - *provided* the ensemble
actually disagrees on the optimistic predictions (being checked via new `wm/disagreement` /
`wm/reward_opt`). **Confirmed usable:** failed seeds have ~50% higher ensemble disagreement
(0.032 vs 0.021) and less-pessimistic reward_opt (-0.005 vs -0.046) than converged. Testing the
penalty (`PMDC_PESSIMISM=5`).

### F8 — The optimism is SYSTEMATIC (ensemble agrees), so disagreement-pessimism does NOT fix it
Tested an ensemble-disagreement pessimism penalty (`reward -= beta * ensemble_spread`):
- 1-step spread, beta=5: **3/5** (no gain).
- compounding horizon spread (per-member 24-step trajectories), beta=0.3: **2/5** (no gain).

Reason: the trained ensemble **agrees even over the horizon** - late disagreement ~0.03 for BOTH
converged (0.024) and failed (0.035) seeds. The world model is **confidently, systematically
optimistic** (all members make the *same* error), not uncertain - so disagreement is the wrong
signal; there is nothing for it to flag.

**Mechanism (best understanding):** PMDC's reward uses `future_state`, the delay-step-ahead
prediction recalibrated by SBSP. SBSP corrects the forward prediction by the CURRENT 1-step residual
(a constant-drift assumption). For on-distribution (good) policies the error is ~flat over the
horizon and the correction works -> accurate reward. For off-distribution (bad) policies the
prediction error GROWS over the 24-step horizon, the constant-drift correction under-corrects ->
`future_state` (and the reward) is optimistic -> SAC is told the bad policy is fine -> no gradient
out of the bad optimum (F6/F7). This is a property of the **SBSP recalibration under distribution
shift**, not the ensemble or its LR.

---

## 3. Resolved: world-model LR is not the lever

E0 (F4) answers the opening question — the ensemble is well-calibrated even in failing seeds, so
the world-model LR / training cadence is not what separates success from failure. The investigation
has moved to the **SAC side** and **reward stationarity**.

---

## 4. Experiment log

(updated as runs complete; see also `/tmp/wm_progress.log`)

- **E0 (baseline WM dynamics):** seeds 0-4, `FIX_PREVOBS=1`, baseline LR, instrumented.
  Result: 1/5 converged (s0=5.6); WM calibrates identically in all seeds. **=> F4: WM fine, SAC
  is the unstable part.**
- **E1 (freeze WM @20k):** prev_obs fix + `FREEZE_WM=20000` -> **2/5** (vs 1/5 baseline). Marginal
  / within-noise; frozen model less accurate (pred_err ~0.8 vs 0.4). Non-stationarity is a minor
  factor. SAC-side metrics then revealed **F5 (critic divergence)**.
- **E2 (SAC lr=1e-4):** prev_obs fix + `SAC_LR=1e-4` -> **0/5**. Lower LR DID stop critic
  divergence (critic_loss bounded 0.55-0.77 vs baseline 2-2.5) but the policy is **under-trained**
  (40-70cm) -> too slow for 60k steps. Insight: critic divergence is a *symptom*; slowing the
  critic just trades divergence for under-training. Need stability WITHOUT slowing learning.
- **E3 (SAC gamma=0.95):** prev_obs fix + `SAC_GAMMA=0.95` -> **2/5**. Critic loss collapses to
  ~0.13 in ALL seeds (vs 1-2.5 baseline) = fully stabilised, but failed seeds stuck in stable poor
  optima (40-89 cm). **=> F6: it's a local-optima / exploration problem, not critic stability.**
- **E4 (gamma + entropy=-1):** prev_obs + `SAC_GAMMA=0.95` + `SAC_TENT=-1` -> **3/5** (best so far;
  s2, which failed under gamma-alone in E3, now converges to 6.5 cm). Confirms F6: **more
  exploration escapes some of the poor optima.** (ent_coef still collapses ~0.01 -> -1 not binding.)
- **E5 (stronger entropy tent=0):** prev_obs + gamma=0.95 + `SAC_TENT=0` -> **1/5 (WORSE)**. Too
  much exploration -> policy too random to track precisely (E4's converged s0,s1 now fail at
  30-58 cm). **Entropy has a sweet spot ~ -1**; pushing further hurts. **Best config = E4 (3/5).**
- **E6 (best config + longer training):** prev_obs + gamma=0.95 + tent=-1 + `STEPS=100000` ->
  tests whether the late breakthrough converts more failed seeds with more compute. **[running]**
- New knobs added: `PMDC_SAC_ENT`, `PMDC_SAC_TENT`, `STEPS`.

---

## 4b. Convergence summary (PMDC @250-290 ms, 5 seeds, all with prev_obs fix unless noted)

| Exp | Config added | converged | takeaway |
|---|---|---|---|
| baseline | original code (no prev_obs fix) | 2/5 | the published instability |
| E0 | prev_obs fix only | 1/5 | WM calibrates fine in ALL seeds (F4) |
| E1 | + freeze WM @20k | 2/5 | reward non-stationarity is minor |
| E2 | + SAC lr 1e-4 | 0/5 | stops critic divergence but under-trains |
| E3 | + gamma 0.95 | 2/5 | critic loss -> ~0.13 (clean); stuck optima (F6) |
| E4 | + gamma 0.95 + target-entropy -1 | **3/5** | exploration escapes some optima |
| E5 | + gamma 0.95 + target-entropy 0 | 1/5 | too random -> hurts; entropy sweet spot ~ -1 |
| E7 | + 1-step disagreement pessimism (b=5) | 3/5 | no gain; 1-step spread too small |
| E8 | + compounding-horizon pessimism (b=0.3) | 2/5 | no gain; ensemble AGREES late (F8) |

**Best config = prev_obs + gamma=0.95 + target-entropy=-1 (E4), ~3/5 (noisy 2-3/5).** Nothing
tested reaches 5/5; the residual failures are the systematic-optimism problem (F7/F8), which no
hyperparameter touches.

Convergence = final tracking err < 15 cm (PMDC lands at ~3-8 cm when it works or 40-90 cm when it
fails; the gap is wide so the threshold is unambiguous). Rates are noisy at n=5.

## 5. Recommendation (current)

1. **World-model learning rate is NOT the issue** (F4) — the ensemble calibrates to pred_err ~0.4
   in every seed, converged or failed. The "under-trained model" guess was wrong (this answers the
   question that kicked off the investigation).
2. **Fix the `prev_obs` bug** (F1) regardless — a genuine world-model-training correctness bug
   (frozen input state; also affects the published FetchPush results); it rescues some seeds alone.
3. **gamma=0.95 makes training clean** (F6) — collapses the critic loss ~10x and removes the
   divergence; sensible as a default even though it doesn't raise the rate by itself.
4. **Operating config: prev_obs + gamma=0.95 + target_entropy=-1.** Cleanest training (critic loss
   ~0.13) and the best rate found (~3/5). gamma + entropy are the only levers that helped; lr,
   buffer, freeze, and disagreement-pessimism did not.
5. **Reliable 5/5 was NOT achieved and is not a hyperparameter problem.** The residual failures are
   SAC stuck in poor optima where PMDC's reward is **systematically optimistic** (F7/F8) - a
   property of the SBSP recalibration under distribution shift, not the WM LR and not ensemble
   uncertainty.
   - **Practical (for the thesis now):** report multiple seeds; the converged runs reach ~5 cm (the
     real PMDC result) vs A-SAC ~25 / SAC ~34. State the SAC training variance honestly.
   - **Research fix (future work):** a less-optimistic reward - a multi-step recalibration that does
     not assume constant drift, or a reward not built from the forward prediction - is the
     principled path to reliable convergence.

---

## 6. Code changes & how to reproduce

All changes are **env-var-gated and non-breaking** (defaults reproduce the original behaviour).

**`PMDC_wrapper.py`** - instrumentation (`self.wm_loss`, `self.wm_pred_err`, logged to TB) + knobs:
- `PMDC_FIX_PREVOBS=1` - update `prev_obs` each step (fixes the F1 data bug).
- `PMDC_FREEZE_WM=N` - stop world-model updates after global step N (stationary reward; F-E1).
- `PMDC_TRAIN_EVERY=K` - additionally train the WM every K steps (0=off; K=1 *hurt*, see F2).
- `PMDC_WM_LR=x` - override ensemble Adam LR (default 5e-5; *not* useful, see F4).

**`delay_correcting_training.py`** - logs `wm/ensemble_loss`, `wm/pred_err`; SAC knobs:
`PMDC_SAC_GAMMA` (0.99), `PMDC_SAC_LR` (3e-4), `PMDC_SAC_BUFFER` (20000), `PMDC_SAC_LSTART`
(20000), `PMDC_SAC_ENT` (auto), `PMDC_SAC_TENT` (auto); `STEPS` via env (driver).

**Recommended config (best found - 3/5 and clean training):**
```bash
PMDC_FIX_PREVOBS=1 PMDC_SAC_GAMMA=0.95 PMDC_SAC_TENT=-1 \
  python3 delay_correcting_training.py --env-id FetchSlide-RemotePDNorm-v0 \
  --delay-ranges 250-290 --algorithms PMDC --steps 60000 --seed <S> --device cuda \
  --operator-model operator_models/FetchSlide-v2_SAC_seed0.zip --output-dir runs_best
```

**Helper scripts** (in `/tmp`, recreate if cleared): `wm_run.sh` (parallel multi-seed driver,
sets all knobs incl. `ALG`/`DELAY`/`OUTDIR`), `wm_analyze.py <tag>` (per-seed err + WM dynamics).

---

## 7. Cross-delay comparison + overnight plan (low / medium / high)

Goal: IQM (line) + IQR (shaded) **training curves** for PMDC vs A-SAC vs SAC at three delays, then
tune PMDC per delay to beat the baselines.

- **Delays:** low 90-120 ms (act 8), **medium 170-200 ms (act 16, chosen midpoint)**, high 250-290 ms (act 24).
- **Config:** A-SAC/SAC = baselines at default SAC (gamma 0.99, ent auto); PMDC = optimised
  (prev_obs + gamma 0.95 + target_entropy -1). Gamma-matched baseline check = Phase-3 option.
- **Long-delay (250-290) result already in (5 seeds):** by IQM PMDC 21.5 ~ A-SAC 22.8 < SAC 32.4;
  by **median** PMDC 8.3 << A-SAC 20.7 < SAC 32.6. The 3/5 rate erases the IQM gap (one failed seed
  leaks into the middle-3); at 4/5 the IQM flips to a clean PMDC win.
- **Pipeline:** (1) 90-120 PMDC -> low+high plots+table; (2) medium driver (PMDC+A-SAC+SAC at
  170-200) -> medium plot + 3x3 table; (3) per-delay PMDC tuning + 10-seed robustness.
- **Plots:** `python3 plot_iqm_curves.py <delay> <pmdc_dir> plots/iqm_<delay>.png` (A-SAC/SAC auto
  from runs_FetchSlide). PMDC dirs: low=runs_FetchSlide_best9012, med=runs_FetchSlide_med_pmdc,
  high=runs_FetchSlide_sac_g95_tent1.

**Results so far (5 seeds, err cm, IQM / median):**

| delay | PMDC (opt) IQM/med | A-SAC | SAC |
|---|---|---|---|
| 90-120 (low) | 6.6 / 3.9 | **3.7 / 3.5** | 4.3 / 4.1 |
| 170-200 (medium) | 13.0 / **8.7** | 10.7 / 10.8 | 20.4 / 20.8 |
| 250-290 (high) | 21.5 / **8.3** | 22.8 / 20.7 | 32.4 / 32.6 |

**Key result (by median / converged performance):** PMDC's converged error is ~FLAT across delays
(3.9 -> 8.7 -> 8.3 cm) because it corrects the delay, while the baselines DEGRADE with delay (A-SAC
3.5 -> 10.8 -> 20.7; SAC 4.1 -> 20.8 -> 32.6). **PMDC's advantage GROWS with delay**: it loses at
low, crosses over A-SAC at the medium delay (median 8.7 < 10.8), and dominates at high (8.3 << 20.7).
This is the thesis claim, confirmed. Plots: `plots/iqm_{90-120,170-200,250-290}ms.png`.

**Caveat:** the 2/5 instability inflates PMDC's IQM/mean at every delay -> by IQM PMDC only TIES
A-SAC; it wins clearly only by median. **Phase 3: lift convergence to ~75%+ so the IQM also flips.**

### Phase 3 - PMDC rate-tuning at high delay (250-290 ms). Baseline best (gamma0.95+tent-1, buf20k) = 3/5, IQM 21.5.
| config tried | conv/5 | IQM | note |
|---|---|---|---|
| + SAC buffer 100k | 2/5 | 45.2 | WORSE - retains early optimistic-reward transitions |
| + real-reward-mix 0.5 | 1/5 | 41.9 | WORSE - delayed real reward adds 24-step credit-assignment noise (the delay PMDC avoids) |
| 10-seed robustness (best cfg) | _running_ | | true rate over 10 seeds |

**Phase 3 conclusion:** NO config beats the best (gamma0.95 + tent-1, ~3/5). Tested: prev_obs,
freeze, lr, gamma, entropy sweep, pessimism (1-step + compounding), buffer, longer training,
real-reward-mix. The ~3/5 (~50-60%) rate is the **ceiling for hyperparameter/reward-shaping levers**;
lifting it needs the fundamental fix (multi-step SBSP recalibration that does not assume constant
drift; F8). PMDC's win is by **median** (converged error flat across delay), with the IQM capped by
the instability.
