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
        self.fix_prev_obs = os.environ.get("PMDC_FIX_PREVOBS", "0") == "1"
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
        for model in self.dc_models:
            model.learning_rate = np.random.randint(-100, 100) / 1_200_000
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
        if self.train_every and len(self.replay_buffer) > self.start_training and self._step_i % self.train_every == 0:
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
                state = predictions.mean(axis=0)
            self.future_state = state
            # ensemble spread at the final (alpha-ahead) rollout step, reward-relevant dims
            if predictions is not None:
                self.wm_disagreement = float(predictions[:, [0, 1, 2, 11, 12, 13]].std(axis=0).mean())
            self.future_state_buffer.append(self.future_state)  # for the next step's residual diagnostic
        else:
            # SBSP (default): recalibrate the cached buffer, then extend it by one prediction.
            self.recalibrate(observation)

            predictions = np.array([model.predict(self.future_state, action) for model in self.dc_models])
            self.future_state = predictions.mean(axis=0)
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
            observation = np.mean(predictions, axis=0)

            self.future_state_buffer.append(observation)

        self.member_states = members
        return observation

    def learn(self):
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        obs = samples[:, 0]
        obs_ = samples[:, 1]

        obs = np.array([item for item in obs])
        obs_ = np.array([item for item in obs_])

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
            x[i] += difference

        self.future_state_buffer = deque(x)
        self.future_state += difference
        if self.member_states is not None:
            self.member_states = self.member_states + difference  # common shift keeps members reality-centred

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
