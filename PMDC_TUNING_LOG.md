# PMDC fair-comparison tuning campaign — running log

**Started:** 2026-06-09 ~18:30 (autonomous session; Luc away).
**Goal:** optimise PMDC's tracking + convergence rate **changing only PMDC code** (wrapper /
world model / reward construction). **Fairness rule: SAC, Adam, and all SAC hyperparameters
stay at the SAME defaults as the A-SAC and SAC baselines** (gamma 0.99, lr 3e-4, ent auto,
buffer 20k, learning_starts 20k). The earlier "stability config" (gamma 0.95, tent −1) tweaked
SAC and is therefore **disallowed** for new arms; it is kept only as a reference.

**Benchmark:** FetchSlide-RemotePDNorm-v0 @ 250-290 ms (act delay α=24), 80k steps,
operator = 67% FetchSlide expert. Metric = `track/step_reward` (true tracking error), final
100 logged episodes, in cm. Convergence threshold 15 cm (bimodal: ~3-13 converged vs ~40-100
failed). Aggregates: IQM + median over seeds. Analyzer: `python3 tuning_analyze.py [--md]`.

---

## Running results table

(updated on every run completion; per-seed = completed seeds only; conv = err < 15 cm.
NB the conv threshold suits PMDC's bimodality; A-SAC is unimodal ~13-26 so read its IQM.)

| arm | fair SAC? | conv | IQM cm | median cm | per-seed cm | config |
|---|---|---|---|---|---|---|
| A-SAC (baseline) | yes | 2/6 | **16.8** | 16.2 | 15.6, 13.5, 20.3, 16.7, 14.8, 25.8 | default SAC on augmented obs (dim 78) |
| SAC (baseline) | yes | 0/5 | 32.0 | 31.4 | 31.0, 29.9, 31.4, 33.7, 43.4 | default SAC on raw delayed obs |
| SBSP-g95 (reference) | **NO** | 1/7 | 41.0 | 48.8 | 48.8, 3.5, 22.7, 55.1, 49.1, 66.4, 29.3 | prevobs + SAC gamma .95, tent −1 (s6 failed LATE: 15.7@63k→29.3@80k) |
| ABSP-g95 (reference) | **NO** | 3/5 | 19.3 | 13.6 | 6.5, 13.6, 69.0, 7.4, 37.0 | as SBSP-g95 + absp (COMPLETE; ~7× compute/step) |
| W1 WMTUNE | yes | 1/5 | 51.4 | 51.8 | 12.3, 51.8, 89.7, 53.1, 49.4 | `FIX_PREVOBS + WM_NORM + WM_LR=1e-3 + CLIP_STATE=10` |
| W2 WMDELTA | yes | 2/5 | 33.4 | 42.3 | **6.4***, **6.6***, 75.9, 51.2, 42.3 | W1 + `PMDC_WM_DELTA=1` (best converged errors of any fair arm) |
| W3 OBS-SCALE | yes | 0/5 | 40.6 | 38.0 | 37.1, 33.2, 69.8, 46.8, 38.0 | W2 + `PMDC_OBS_ACTSCALE=80` — hurts (see below) |
| **W4 WM-WARMUP** | yes | **6/10** | **15.0** | **8.1** | 4.7*, 9.2*, 97.3, 5.4*, 7.0*, 5.9*, 45.5, 49.8, 5.9*, 16.2 | **W2** + `PMDC_WM_WARMUP=10000` — winner; n=10 GATE PASSED (median 8.1 = 2× A-SAC's 16.2; IQM 15.0 < 16.8) |

## Thesis grid campaign (launched 14:58, Luc-directed)

Full grid per Luc's reference figure: (a-c) SBSP vs ABSP, (d-f) PMDC/A-SAC/SAC at
**90-130 / 170-210 / 250-290 ms**, (g-i) 3D EE trajectories @250-290. All PMDC arms = W4 fair
config (ABSP adds `PMDC_PRED_MODE=absp`); SAC defaults everywhere; 80k; n=5/new cell (n=10
SBSP@250-290 reused). 45 new runs via `thesis_grid_master.sh` (idempotent), critical path
fair-ABSP@250-290 ×5 first (~7 h each). ETA full grid ≈ 28-30 h (→ ~2026-06-11 evening).
Figure: `thesis_results_grid.py` (renders partial grids; rerun anytime). Trajectories:
`capture_trajectories.py` (running; PMDC WM warmed online per Pitfall E, median-error episode
plotted for honesty).

**W4 final (11:25): 4/5 converged, IQM 7.2 cm, median 7.0 — best arm of the campaign,
including the SAC-tweaked references, at fully default SAC.** 2.3× better IQM than A-SAC
(16.8). Mechanism held: dense WM warmup (first 10k steps) → reward stationary before
learning_starts → earliest breakthroughs of any arm (s3 at 5 cm by 30k); even s0 (73 cm at
59k) recovered with a late breakthrough to 4.7. The α-step residual is uniform 0.07-0.09 and
optimism ~0 in every seed. s2 still hard-failed (97 cm) with a *perfect* WM and the same
critic level as the converged seeds — the F6 seed-lottery residue persists at ~1/5. Converged
critics run hot (13-16, one 621 spike) yet converge — at gamma 0.99 the absolute critic level
is evidently not the failure signal (revising F5's relevance to fair-config runs).
Caveat: n=5 and rates are noisy (cf. SBSP-g95 3/5@60k → 1/7@80k) → seeds 5-9 launched 11:30
for n=10 before declaring; W4 config = `PMDC_FIX_PREVOBS=1 PMDC_WM_NORM=1 PMDC_WM_LR=1e-3
PMDC_CLIP_STATE=10 PMDC_WM_DELTA=1 PMDC_WM_WARMUP=10000`.

**W3 analysis (08:00): a clean, informative negative.** Critic losses did drop (final 5-11 for
4/5 seeds vs W2's 22-68) — the conditioning worked — but 0/5 converged. The covariate that
flipped: **entropy fell to 0.04-0.07 (W2: 0.10-0.27)**. Better-conditioned inputs let SAC fit
faster → auto-entropy anneals sooner → less exploration → the F6 breakthrough never
consolidates (s1 reached 23 cm @30-40k and regressed; s4 27 cm @30k, drifted back).
**Synthesis across W1-W3 + E4/E5: at default SAC the breakthrough is EXPLORATION-gated; W2's
early reward-pinning (flat clipped reward → auto-entropy stays high) was accidentally doing
what target_entropy=−1 did for the legacy config.** W4 tests the flip side: WM warmup makes
the early reward stationary (removing W2's s2-style critic-457 divergence risk) at the cost of
the accidental exploration phase — the readout adjudicates stationarity vs exploration.

**08:20 — seed-level check WEAKENS the exploration synthesis (recorded for honesty):** across
all 15 fair PMDC seeds, corr(entropy@20-50k, final err) = −0.10 (Spearman −0.19). The
arm-level ordering (W2 > W3 entropy and W2 > W3 convergence) holds, but within arms entropy
does NOT predict which seed converges (W2's most-exploring seed is its worst). So exploration
level is not the seed-selector either. Running tally of ELIMINATED causes for the seed
lottery: WM accuracy (W1), reward optimism (W1/W2, wm_opt ≤ 0 everywhere), input conditioning
(W3), entropy level (n=15 corr ≈ 0). Remaining live candidate: early reward/MDP
non-stationarity (W4, in flight); else the residual is SAC's optimisation landscape on the
induced MDP (F6) — a finding in itself for the chapter.

**W2 analysis (04:40):** delta WM nails the prediction axis (α-step residual 0.12-0.30 — best
of any arm incl. legacy; optimism → 0) and produces the best converged errors (6.4/6.6 cm @
critic ~25!). But critic losses are huge across all seeds (final 22-68, peaks 90-457): the
untrained delta WM random-walks early rollouts → predicted tracking distance ~2 m → **reward
pinned at the −2 clip for ~20k steps, then migrates to −0.05** — a large reward shift sitting
in SAC's first-update buffer. Side effect: pinned-flat early reward keeps auto-entropy high
(0.10-0.27 vs W1 0.005-0.05) = accidental exploration, plausibly why delta converges better
anyway. ⇒ W3 attacks critic input conditioning; W4 candidate attacks early reward
stationarity (dense WM warmup so the reward converges before SAC's buffer fills).

Historical anchors (60k, original code, FetchSlide @250-290, from PMDC_STABILITY_INVESTIGATION.md):
PMDC original **2/5** conv (5.0, 4.7, 43.8, 47.7, 54.1); A-SAC 5/5 (~20-36); SAC 5/5 (~27-47).
Legacy stability config (SAC-tweaked): 3/5 @60k but **1/6 in the fresh 80k batch** — the
convergence rate is genuinely noisy run-to-run; n=5 rates carry ±1-2 seeds of noise.
Target: fair-SAC PMDC with conv ≥ 3/5 and converged error ~5-13 cm ⇒ beats A-SAC's 16.8 IQM.

---

## Why W1 = world-model input-norm + lr 1e-3 (evidence)

1. `wm_hparam_ablation.py` (FetchPush, deployment WM budget ~1600 updates, held-out one-step
   remote-EE error): default lr 5e-5 → **15.7 cm**; lr 1e-3 → 12.1; **lr 1e-3 + z-score input
   norm → 4.2 cm (3.7×)**. Mechanism: PD-gain action (±80, std ≈ 46) swamps the ~±1 state
   features in the un-normalised first layer.
2. F7/F8 (investigation): residual instability = world-model-derived reward is *systematically
   optimistic* for off-distribution policies → SAC stuck in poor optima. A 3.7× more accurate
   WM shrinks the residuals SBSP's constant-drift patch must correct → less optimism.
3. `PMDC_FIX_PREVOBS=1` fixes the frozen-`prev_obs` data bug (F1) — pure correctness, WM-side.

Hypothesis under test: better WM (PMDC-side only) lifts the convergence rate at **default SAC**,
where the original code managed 2/5 (and prevobs-only 1/5, E0).

## Wave 2 levers (implemented, gated, validated offline before any full run)

- `PMDC_WM_DELTA=1` — WM predicts s_{t+1}−s_t (residual dynamics; standard parameterisation).
- `PMDC_RECAL_GROWTH=λ` — horizon-scaled SBSP recalibration: entry i+1 steps ahead patched by
  `residual·(1+λ(i+1))` instead of constant residual. Directly targets F8's under-correction.
- Reserve: `PMDC_FREEZE_WM` with the *accurate* WM (E1 retest), WM buffer/batch size.

---

## Decisions log

- **2026-06-09 18:30 — GPU reallocation.** Master ABSP campaign (absp80k_master.sh) stopped
  dispatching (dispatcher killed; **idempotent — rerun the script to resume seeds 6-7 later**).
  Rationale: Luc's new directive prioritises fair-SAC PMDC tuning; the master's PMDC arms use
  the now-disallowed SAC tweaks, and its baselines already have 5-6 completed seeds. Kept
  in-flight absp_s3/s4 (~7.6 h sunk, nearly done). Killed <35-min-old absp_s5, sbsp_s6, asac_s6
  (arms already at n=5-6; rerunnable). All 5 slots → W1 WMTUNE seeds 0-4 in parallel (~4-4.5 h).
- Campaign infra: `pmdc_arm_driver.sh` (generic idempotent arm driver), `tuning_watch.sh`
  (re-invokes analysis on each completion), `tuning_analyze.py` (per-seed instability metrics:
  critic loss, ent_coef, wm/pred_err, wm/reward_opt optimism, wm/ensemble_loss, wm/disagreement).

## W1 readout (01:20) — the hypothesis is dead; long live the next one

**1/5 converged (s0=12.3 cm; IQM 51.4).** The tuned WM did exactly what it promised — wm_loss
20× better than legacy, `wm/reward_opt` ≤ 0 in every seed and every 10k window (the F7
optimism is GONE) — and convergence did not move (original code 2/5, E0 1/5, W1 1/5).
**WM accuracy was never the binding constraint at default SAC; optimism was a covariate, not
the cause.** F6 is the standing diagnosis: SAC at gamma 0.99 gets stuck/unstable on this MDP.

Per-seed failure modes (10k-window traces in tuning_analyze):
- s0 *: clean late breakthrough 50-60k (Pitfall-I pattern), critic ≤ 1.5.
- s1: **broke through to 17 cm @60-70k then REGRESSED to 34** as critic climbed 2.7 → 7.2 —
  late critic destabilisation, not a stuck policy.
- s2: progressive worsening 41→83 with the BEST WM of the batch (wm_err 0.79) — honest reward,
  policy still degraded; critic creeping 0.6→2.7.
- s3: flat-stuck 43-60 all run, critic LOW (≤1.4) — textbook F6 stable poor optimum.
- s4: critic explosion from 30k (13–38), entropy highest (0.05), reward strongly pessimistic
  (wm_opt −1.8), worst WM (wm_err 3.4) — SAC divergence dragging the loop.

⇒ The actionable signal is **critic conditioning at default gamma**: 3/5 failures involve a
hot critic (vs 0.05-0.24 in the g95 reference). PMDC cannot touch gamma (fairness), but it CAN
fix what it feeds the critic: the augmented obs mixes O(1) state dims with **O(80)
action-history dims** — the same scale pathology that crippled the WM (and plausibly why
A-SAC's critic runs at 1.7-7.7 too). **W3 = scale the action-history dims by 1/80**
(`PMDC_OBS_ACTSCALE=80`, PMDC-branch-only interface change; baselines untouched). Smoke-tested.

Secondary observation: W1's final α-step residual (wm_err 0.8-3.4) is *higher* than legacy's
~0.4 despite the better one-step WM — consistent with W2's delta-WM being the right next lever
on that axis (live smoke: residual 4.2 vs 23 at 3k steps).

## Wave-2 decision tree (mechanical, applied at each watcher wake)

Gates first (run via `orchestrate_gates.sh` once GPU load allows): Gate A = delta WM
(`delta_validation.py`); Gate B = recal-growth λ (`recal_growth_validation.py`).

- **W1 conv ≥ 4/5** → WM accuracy was the binding constraint. Extend WMTUNE to seeds 5-7;
  stack the passing gate levers as W2 for a possible further win.
- **W1 conv 2-3/5** → partial. Mine failed seeds: high `wm/reward_opt` → prioritise
  recal-growth (W2 = wmtune + λ*); high `wm/pred_err` → delta / WM capacity; critic divergence
  with accurate WM → reward-side levers.
- **W1 conv ≤ 1/5** → WM accuracy not binding at default SAC (would echo F4). Pivot to reward
  construction: recal-growth (less optimistic reward), then reconsider buffer/freeze with the
  accurate WM (E1 retest).

Convergence bar: original code 2/5 @60k; prevobs-only 1/5 @60k (E0); legacy SAC-tweaked best
3/5 @60k (disallowed now). A fair-SAC arm at ≥3/5 with converged-seed error ~5-13 cm already
beats every fair predecessor.

## W1 first-launch crash post-mortem (21:02) — three bugs, all fixed

All 5 W1 seeds died with NaN actor output at step ≈20,000 = `learning_starts` (the first time
SAC's networks run; random sampling before that hides everything). TB forensics: tracking
stayed sane but `wm/pred_err` was NaN/inf from the **first WM `learn()` onward** (finite for
exactly the ~21 pre-learn episodes). Chain of three:

1. **Mine — degenerate z-score std.** `std + 1e-6` on near-constant FetchSlide obs dims (a
   resting puck) → the ITERATED rollout divides its own output noise by ~1e-6 → 1e6×
   amplification per step → inf by rollout step ~3 → NaN absorbs the SBSP buffer → SAC's
   replay fills with NaN obs → first actor query crashes. One forward pass on real data never
   triggers it (deviation ≈ 0), which is why the offline ablation + smoke passed.
   **Fix: absolute std floor 1e-2.**
2. **Original code — pre-training rollout explosion.** Untrained-ensemble α-step rollouts
   occasionally explode (~1e16); those states reach SAC as observations (and, pre-fix, the
   WM replay via the reset-time `prev_obs` remnant). Legacy runs survived by an accident:
   the F1 bug trained the WM on its own exploded states, crushing the weights back down
   (self-damping). Fixing F1 removed that accidental protection.
   **Fix: `PMDC_CLIP_STATE=10`** — predicted states clipped to a generous workspace box
   (real obs are O(1); never binds in normal operation) + NaN→num.
3. **Original code — aliasing: emitted observations mutated after delivery.**
   `recalibrate()` did `future_state += difference` IN PLACE on the same ndarray emitted as
   last step's observation — which the outer delay wrapper still holds awaiting delayed
   delivery. Every PMDC run ever has delivered recalibration-shifted obs through this side
   channel (small once converged, chaotic early). **Fix: rebind instead of in-place add**
   (value-identical at defaults; only the side effect is removed).

Verification: 3000-step strict smoke — obs stream finite, `worst_state=10.0000`,
`in_std min=0.01`, zero non-finite `wm/pred_err` after learns. Relaunched 21:57.

Also: **Gate A (delta prediction) PASSED — 2.60 cm vs 4.02 cm one-step** (FetchSlide,
deployment budget; also replicates the FetchPush norm+lr result on this env). `PMDC_WM_DELTA=1`
is cleared for wave 2.

**Gate B (recal-growth λ) FAILED — decisively (22:14).** Frozen tuned WM, ON=trained policy /
OFF=random: λ=0 gives pred_err 0.47/0.58 with reward optimism already ≈ −0.1 (pessimistic!,
p90 +0.05); every λ>0 monotonically degrades pred_err (0.88 @λ=.1, 1.49 @λ=.3, 10.8 @λ=1) and
over-corrects into stronger pessimism. Two conclusions: (1) the constant-drift patch is the
better estimator once the WM is accurate — drop recal-growth (knob stays, default 0);
(2) **the tuned WM appears to remove the F7/F8 optimism outright** (legacy weak-WM probes
showed +0.2..0.5 optimism in failures; tuned WM probe shows none even off-distribution,
warmup wm_loss 2e-4 vs legacy 1.7e-2). This is the W1 mechanism working as hypothesised —
pending the W1 tracking readout to confirm it transfers to the coupled training loop.
⇒ **Wave 2 = W1 config + PMDC_WM_DELTA=1** (single added lever, launch after W1 readout).

## Reboot #2 (2026-06-10 evening) + recovery (06-11 09:55)

PC restarted before ~22:00 on 06-10: lost the in-flight matched-W4 runs, all five fair-ABSP
@250-290 runs (~3.5 h each), and ~12 h of idle overnight. Nothing on-distribution was
persisted. Relaunched 09:55: matched-W4 ×3 (PMDC_SAVE_WM=1, priority — examiner-ready
on-distribution figures ~14:30-15:00) + grid master at 2 slots, auto-bumped to 5 when the
matched runs land (dynamic `/tmp/pmdc_grid_slots`). Revised grid ETA: absp250 by ~20:00,
full grid ~06-12 morning. The warmed-replay WM-accuracy figures (FetchSlide + FetchPush,
`diagnostics/wm_accuracy*/`) survived on disk, as did all completed training runs.

## FetchReach status (2026-06-11 16:30) + plan

FetchReach has the env port, a 100%-success operator, and `slurm/run_pmdc_comparison.sbatch`
— but was NEVER run (the wandb project doesn't exist; no local runs). The sbatch carried the
old SAC-tweaked config; **updated to the W4 fair config + PMDC_SAVE_WM** (submit-ready:
`sbatch --array=0-31%8 slurm/run_pmdc_comparison.sbatch`, override DELAY for other settings).
**Default plan: when the FetchSlide grid completes (~06-12 AM), queue the FetchReach grid
locally** (3 delays × 4 methods × 5 seeds, ~30 h → ~06-13), unless Luc submits to SLURM
first. FetchPush on-distribution matched runs in flight (3 seeds, done ~19:00 → figures).

## On-distribution WM-accuracy results (matched pairs, 2026-06-11 ~15:30)

Matched (policy, WM) pairs persisted + reloaded (3 converged reruns: 7.3/7.7/6.8 cm; loader
verified: one-step remote-EE 3.1 cm median on fresh transitions). Figures in
`diagnostics/wm_accuracy_ondist/` (absolute remote-EE, reward-relevant track, drift).

**Findings (FetchSlide @250-290, α=24):**
1. The "prediction for t+k available at t" genuinely compounds even on-distribution:
   absolute remote-EE 4.2→32.1 cm (k=1→24); reward-relevant |pred_dist−true_dist|
   2.5→21.8 cm. ABSP is better (track 1.6→15.9) at 24× the per-step compute;
   no-recal worst (23.1 @k=24). **Luc's reading is partially right on-distribution: SBSP's
   instantaneous forward estimate is noisier than ABSP's at long horizons.**
2. BUT the noise is **zero-mean**: episode-mean predicted tracking ≈ true within ~3 cm
   (frozen pair: pred 28.8 vs true 26.0), and the residual at realisation (after the entry's
   full patch history) is 0.075 full-state L2 = deployed level. The reward stream SAC trains
   on is unbiased; that, plus dense re-anchoring, is why PMDC trains to 7-8 cm despite the
   noisy instantaneous estimates. Thesis framing: plot BOTH curves; pair with the downstream
   result (SBSP-PMDC ≈ ABSP-PMDC tracking at 1/α compute, from the grid).
3. **New phenomenon — eval gap:** reloaded matched policies track at ~19-26 cm (deterministic)
   vs 7-8 cm training-tail, with honest predictions and deployed-level residuals; a 2k-step
   WM touch-up does NOT fix it; stochastic eval is worse (34.5). The gap is policy-side
   (training-process effect / late-training churn), NOT WM-side — echoes the operator
   finding (final checkpoint ≠ training-tail performance; cf. promote-best-not-final).
   Per-episode eval medians are good (capture: 4.6 cm median) with heavy tails. For the
   thesis: report training-tail metrics (as Table 2 does) and note snapshot-eval caveat,
   or add best-checkpoint saving to PMDC training as future work.

## Per-run instability analysis

**80k batch, final-window metrics (2026-06-09 20:45):**
- **SBSP-g95 failed seeds (s0,s2,s3,s4,s5)** end with *calibrated* WMs (`wm/pred_err` 0.40-0.49
  vs converged s1's 0.33), *near-honest* reward (`wm/reward_opt` −0.06..+0.02) and *low stable*
  critic (0.05-0.24): the F6 stable-poor-optimum signature. The damage is done mid-training
  (F7's optimism during the stuck phase); by the end the loop looks healthy but the policy is
  stuck at 22-66 cm. ⇒ final-window optimism is NOT a sufficient failure detector; need the
  mid-training trace (W1 analysis must look at wm/reward_opt over steps 20-60k).
- **ABSP-g95 failures correlate with strong FINAL optimism** (s2: 69 cm, wm_opt **+0.53**;
  s4 trending 41.9 cm @70k, +0.29) and its delay-horizon error is ~3x SBSP's (wm/pred_err
  ~1.1-1.4 vs ~0.4 — no recalibration, pure open-loop compounding). ABSP trades SBSP's
  under-correction for raw compounding error; both lose.
- **A-SAC** is reliable-but-plateaued (13.5-25.8, critic 1.7-7.7 — high but it still tracks);
  **SAC** uniformly ~30-43. Neither has PMDC's bimodality.
- **First-pair poisoning bug found & fixed before W1**: `initial_undelay` left `prev_obs`
  holding an exploded α-step prediction (~1e16 pre-training) → poisoned each episode's first
  replay pair and, with `PMDC_WM_NORM=1`, the z-score stats (in_std ~2.5e16). Fix: reset
  `prev_obs` to the true reset obs after `initial_undelay` (gated by `PMDC_FIX_PREVOBS`).
  Smoke-verified: in_std_max 46.7 (action dim), buffer max 80.0. This bug also explains a
  long-standing wart: SAC receives garbage obs at early-episode starts pre-WM-training (still
  original behaviour; bounded by the reward clip and self-heals).

**Metric trap (2026-06-09 20:05):** the sparse CSVs (`pred_results/.../seed_N.csv`,
every 100 steps) are PHASE-BIASED: 100 ≡ 0 (mod 50-step episodes), so every sample lands on
an episode-START transient (remote far from operator before the PD loop catches up). A
converged seed reads ~28 cm in the CSV vs 3.5 cm true (TB windowed mean). Never use the CSVs
for final numbers — TB `track/step_reward` only (extends CLAUDE.md Pitfall G).
