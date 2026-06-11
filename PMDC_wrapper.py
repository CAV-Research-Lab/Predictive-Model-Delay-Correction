import gymnasium as gym
import torch
import torch.nn.functional as F
from delay_correcting_nn import DCNN, run
import pickle
from collections import deque
import numpy as np
import os
import sys
import pandas as pd

from csv import writer


class PMDC(gym.Wrapper):
    def __init__(self, env, delay, env_id, pretrain, n_models, act_delay_range=None):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env
        self.delay = delay
        self.env_id = env_id
        self.pretrain = pretrain
        self.n_models = n_models

        self.future_state = None
        self.future_state_buffer = deque()

        self.buffer_size = 10_000
        self.batch_size = 256
        self.start_training = 1000
        self.replay_buffer = deque(maxlen=self.buffer_size)

        # Experimental stabilisation knobs (env-var gated; defaults preserve original behaviour).
        # PMDC_TRAIN_EVERY=K -> also train the world model every K env steps (0 = only once/episode).
        # PMDC_FIX_PREVOBS=1 -> update prev_obs each step so (state, action, next_state) pairs are valid.
        self.train_every = int(os.environ.get("PMDC_TRAIN_EVERY", "0"))
        # PMDC_WM_WARMUP=N -> train the WM every 2 steps for the FIRST N env steps only, then
        # fall back to the default once-per-episode cadence. Goal: the WM (and so the reward
        # function) converges BEFORE learning_starts, so SAC's first updates see a stationary
        # reward. Distinct from PMDC_TRAIN_EVERY (continuous dense updates, which hurt -- F2).
        self.wm_warmup = int(os.environ.get("PMDC_WM_WARMUP", "0"))
        self.fix_prev_obs = os.environ.get("PMDC_FIX_PREVOBS", "0") == "1"
        # PMDC_WM_NORM=1 -> z-score the world-model inputs. The PD-gain action (+-80) is ~50x the
        # state scale and swamps the net; normalising it makes the WM ~3x more accurate at the
        # deployment training budget (see wm_hparam_ablation.py). Default off (non-breaking).
        self.wm_norm = os.environ.get("PMDC_WM_NORM", "0") == "1"
        # PMDC_RECAL_GROWTH=lambda -> horizon-scaled SBSP recalibration: the buffered prediction
        # i+1 steps ahead is patched by difference*(1 + lambda*(i+1)) instead of the constant
        # difference. Targets F8: the constant-drift patch UNDER-corrects when prediction error
        # grows over the horizon (off-distribution optimism). 0 (default) = original behaviour.
        self.recal_growth = float(os.environ.get("PMDC_RECAL_GROWTH", "0"))
        # PMDC_CLIP_STATE=X -> clip every predicted state to [-X, X] and replace NaN/inf.
        # The workspace obs are O(1), so X=10 never binds in normal operation; it only stops
        # the untrained ensemble's iterated alpha-step rollouts from exploding (a rare but real
        # original-code behaviour: those states reach SAC's replay buffer as observations and
        # can NaN the first SAC updates at learning_starts). 0 (default) = original behaviour.
        self.clip_state = float(os.environ.get("PMDC_CLIP_STATE", "0"))
        self._step_i = 0
        self.wm_loss = 0.0       # mean ensemble Huber loss at the last world-model update
        self.wm_pred_err = 0.0   # delay-horizon prediction error ||obs - predicted|| each step
        self.freeze_wm = int(os.environ.get("PMDC_FREEZE_WM", "0"))  # stop WM updates after this global step (0=never)
        self.pessimism = float(os.environ.get("PMDC_PESSIMISM", "0"))  # subtract beta*ensemble_disagreement from reward
        self.wm_disagreement = 0.0
        self.wm_reward_opt = 0.0  # real_dist - predicted_dist; >0 = world model OPTIMISTIC about tracking
        self.member_states = None  # per-member forward trajectories (compounding-uncertainty penalty)
        self.real_reward_mix = float(os.environ.get("PMDC_REAL_REWARD_MIX", "0"))  # blend honest delayed reward

        # PMDC_PRED_MODE selects the delay-corrected state predictor:
        #   "sbsp" (default) -- State-Buffer based State Prediction: reuse a cached buffer and
        #                       correct it with the current 1-step residual (constant-drift recalibration).
        #   "absp"           -- Action-Buffer based State Prediction: recompute the full alpha-step
        #                       rollout from the latest true observation each step (no recalibration).
        # ABSP is the prior method (O(alpha)/step vs SBSP's O(1)); see thesis Ch.3 SBSP-vs-ABSP.
        self.pred_mode = os.environ.get("PMDC_PRED_MODE", "sbsp").lower()
        self.action_buffer = deque(maxlen=self.delay)  # last alpha actions, for ABSP rollouts

        self.layer_size = 128
        self.n_layers = 2

        self.d_reward = 0
        self.d_reward_total = 0
        self.first = False
        self.log = pd.DataFrame({"Step": [], "Delayed episodic reward": []})

        params = self._load_or_create_model_params()
        wm_lr = os.environ.get("PMDC_WM_LR")
        if wm_lr:
            params["beta"] = float(wm_lr)  # override world-model (ensemble) learning rate
        self.dc_models = [DCNN(**params).eval() for _ in range(n_models)]
        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])

        if pretrain:
            for i in range(n_models):
                model_file = f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd_{i}.pt"
                if not os.path.isfile(model_file):
                    if pretrain:
                        print(f"--- Creating Pretrain-Model {i} ---")
                        run(env_id=self.env_id, epochs=4, batch_size=256, file_name=model_file)
                state_dict = torch.load(model_file)
                self.dc_models[i].eval()
                print(f"--- Loaded Model {i} ---")
                self.dc_models[i].load_state_dict(state_dict)

    def _load_or_create_model_params(self):
        params_path = f"./models/{self.env_id}/1_step_params.pickle"
        legacy_params_path = "./models/FetchPush-RemotePDNorm-v0/1_step_params.pickle"
        if os.path.isfile(params_path):
            return pickle.load(open(params_path, "rb"))
        if os.path.isfile(legacy_params_path):
            return pickle.load(open(legacy_params_path, "rb"))

        return {
            "beta": 0.00005,
            "input_dims": self.observation_space.shape[0],
            "n_actions": self.action_space.shape[0],
            "layer_size": self.layer_size,
            "n_layers": self.n_layers,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.d_reward_total = 0

        self.prev_obs = obs
        self.future_state_buffer = deque()
        # Prime the ABSP action buffer with alpha zero-actions, matching initial_undelay's
        # zero-action assumption, so the first steps still roll a full alpha-step horizon.
        zero_act = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.action_buffer = deque([zero_act.copy() for _ in range(self.delay)], maxlen=self.delay)

        self.future_state = self.initial_undelay(obs)
        if self.fix_prev_obs:
            # initial_undelay overwrites prev_obs with PREDICTED states (the F1 bug's reset-time
            # remnant). Before the ensemble is trained those alpha-step rollouts can explode
            # (~1e16), poisoning the episode's first replay pair -- and with PMDC_WM_NORM=1 the
            # z-score stats. The first pair must use the TRUE reset observation.
            self.prev_obs = obs

        if len(self.replay_buffer) > self.start_training and not (self.freeze_wm and self._step_i >= self.freeze_wm):
            self.learn()
        return self.future_state, info

    def step(self, action):
        observation, d_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.d_reward = d_reward
        self.d_reward_total += d_reward

        training_data = (
            np.append(self.prev_obs.astype(np.float32), action.astype(np.float32)).astype(np.float32),
            observation.astype(np.float32),
        )
        self.replay_buffer.append(training_data)

        self._step_i += 1
        _te = self.train_every
        if self.wm_warmup and self._step_i < self.wm_warmup:
            _te = 2  # dense warmup cadence (see PMDC_WM_WARMUP)
        if _te and len(self.replay_buffer) > self.start_training and self._step_i % _te == 0:
            self.learn()
        if self.fix_prev_obs:
            self.prev_obs = observation

        if self.pred_mode == "absp":
            # ABSP: recompute the full alpha-step rollout from the latest TRUE observation,
            # applying the buffered in-flight actions in order. No recalibration / constant-drift
            # patch -- each future state is recalculated from scratch (O(alpha) per step).
            self._measure_residual(observation)  # diagnostics only; keeps the buffer length bounded
            self.action_buffer.append(action.astype(np.float32))
            state = observation.astype(np.float32)
            predictions = None
            for past_action in self.action_buffer:  # oldest -> newest == alpha forward steps
                predictions = np.array([model.predict(state, past_action) for model in self.dc_models])
                state = self._sane(predictions.mean(axis=0))
            self.future_state = state
            # ensemble spread at the final (alpha-ahead) rollout step, reward-relevant dims
            if predictions is not None:
                self.wm_disagreement = float(predictions[:, [0, 1, 2, 11, 12, 13]].std(axis=0).mean())
            self.future_state_buffer.append(self.future_state)  # for the next step's residual diagnostic
        else:
            # SBSP (default): recalibrate the cached buffer, then extend it by one prediction.
            self.recalibrate(observation)

            predictions = np.array([model.predict(self.future_state, action) for model in self.dc_models])
            self.future_state = self._sane(predictions.mean(axis=0))
            # 1-step ensemble spread on the reward-relevant dims (EE pos [0:3] + operator pos [11:14])
            self.wm_disagreement = float(predictions[:, [0, 1, 2, 11, 12, 13]].std(axis=0).mean())

            self.future_state_buffer.append(self.future_state)

        reward = self.calculate_reward(self.future_state)
        if self.pessimism and self.member_states is not None:
            # compounding horizon uncertainty: each member rolls forward on its OWN trajectory,
            # so the spread reflects the model's uncertainty about the delay-step-ahead state.
            self.member_states = np.array([self.dc_models[i].predict(self.member_states[i], action)
                                           for i in range(self.n_models)])
            self.wm_disagreement = float(self.member_states[:, [0, 1, 2, 11, 12, 13]].std(axis=0).mean())
            reward -= self.pessimism * self.wm_disagreement  # penalise uncertain (optimistic) predictions
        if self.real_reward_mix:
            reward = (1.0 - self.real_reward_mix) * reward + self.real_reward_mix * float(self.d_reward)

        return self.future_state, reward, terminated, truncated, {"Delayed Reward": self.d_reward_total}

    def initial_undelay(self, observation):
        action = [0] * self.action_space.shape[0]
        members = np.array([observation] * self.n_models, dtype=np.float32) if self.pessimism else None
        for step in range(self.delay):
            predictions = []
            self.prev_obs = observation

            for i, model in enumerate(self.dc_models):
                predictions.append(model.predict(observation, action))
                if members is not None:
                    members[i] = model.predict(members[i], action)
            observation = self._sane(np.mean(predictions, axis=0))

            self.future_state_buffer.append(observation)

        self.member_states = members
        return observation

    def learn(self):
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        obs = samples[:, 0]
        obs_ = samples[:, 1]

        obs = np.array([item for item in obs])
        obs_ = np.array([item for item in obs_])

        if self.wm_norm:
            # z-score stats from the whole replay buffer; set on each ensemble member's device.
            # float64 accumulation: the +-80 action dim overflows float32 variance on outliers.
            # std FLOOR 1e-2 (not +1e-6): near-constant obs dims (e.g. resting-puck rotations)
            # otherwise divide the ITERATED rollout's own output noise by ~1e-6, amplifying
            # 1e6x per rollout step -> inf within 3 of the alpha=24 steps (the 2026-06-09 W1
            # NaN crash). One forward pass on real data never sees this; the closed loop does.
            all_x = np.array([item[0] for item in self.replay_buffer], dtype=np.float64)
            in_mean = torch.tensor(all_x.mean(axis=0), dtype=torch.float32)
            in_std = torch.tensor(np.maximum(all_x.std(axis=0), 1e-2), dtype=torch.float32)
            for model in self.dc_models:
                model.in_mean = in_mean.to(model.device)
                model.in_std = in_std.to(model.device)

        losses = []
        for model in self.dc_models:
            model.train()
            losses.append(model.learn(obs, obs_))
            model.eval()
        self.wm_loss = float(np.mean(losses))

    def _measure_residual(self, observation):
        """Pop the oldest alpha-ahead prediction and log the realised delay-horizon error.
        Shared by SBSP (which then patches the buffer with it) and ABSP (diagnostics only)."""
        if self.future_state_buffer:
            predicted_state = self.future_state_buffer.popleft()
        else:
            raise Exception("Delay must be greater than 0.")

        difference = observation - predicted_state
        self.wm_pred_err = float(np.linalg.norm(difference))
        _pd = np.linalg.norm(predicted_state[[0, 1, 2]] - predicted_state[[11, 12, 13]])
        _rd = np.linalg.norm(observation[[0, 1, 2]] - observation[[11, 12, 13]])
        self.wm_reward_opt = float(_rd - _pd)  # >0 = predicted error < real error = optimistic reward
        return difference

    def recalibrate(self, observation):
        difference = self._measure_residual(observation)

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            # entry i is the prediction i+1 steps ahead; recal_growth=0 -> constant-drift (original)
            x[i] += difference * (1.0 + self.recal_growth * (i + 1))

        self.future_state_buffer = deque(x)
        # future_state duplicates the deepest buffer entry -> same scale keeps them consistent.
        # REBIND, do not `+=`: the emitted observation is this very ndarray, and the outer
        # delay wrapper still holds it (by reference) in past_observations awaiting delayed
        # delivery -- an in-place update retroactively mutates observations already "sent"
        # (original-code aliasing bug, found 2026-06-09).
        self.future_state = self._sane(self.future_state + difference * (1.0 + self.recal_growth * len(x)))
        if self.member_states is not None:
            # common shift keeps members reality-centred
            self.member_states = self.member_states + difference * (1.0 + self.recal_growth * len(x))

    def _sane(self, state):
        """Bound a predicted state to the (generous) workspace box; see PMDC_CLIP_STATE."""
        if self.clip_state:
            return np.clip(np.nan_to_num(state, nan=0.0, posinf=self.clip_state,
                                         neginf=-self.clip_state),
                           -self.clip_state, self.clip_state)
        return state

    def calculate_reward(self, obs):
        reward = -np.linalg.norm(obs[[0, 1, 2]] - obs[[11, 12, 13]])
        reward = max(-2, reward)
        return float(reward)

    def save_models(self):
        for i in range(self.n_models):
            model_file = f"./models/{self.env_id}/{self.n_layers}-{self.layer_size}-1_step_prediction_sd_{i}.pt"
            if not os.path.isfile(model_file):
                print("Unable to update model!")
                sys.exit()
            else:
                torch.save(self.dc_models[i].state_dict(), model_file)
