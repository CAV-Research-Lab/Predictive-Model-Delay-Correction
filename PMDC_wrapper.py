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

        self.layer_size = 128
        self.n_layers = 2

        self.d_reward = 0
        self.d_reward_total = 0
        self.first = False
        self.log = pd.DataFrame({"Step": [], "Delayed episodic reward": []})

        params = self._load_or_create_model_params()
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

        self.future_state = self.initial_undelay(obs)

        if len(self.replay_buffer) > self.start_training:
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

        self.recalibrate(observation)

        predictions = []
        for model in self.dc_models:
            predictions.append(model.predict(self.future_state, action))
        self.future_state = np.mean(predictions, axis=0)

        self.future_state_buffer.append(self.future_state)

        reward = self.calculate_reward(self.future_state)

        return self.future_state, reward, terminated, truncated, {"Delayed Reward": self.d_reward_total}

    def initial_undelay(self, observation):
        action = [0] * self.action_space.shape[0]
        for step in range(self.delay):
            predictions = []
            self.prev_obs = observation

            for model in self.dc_models:
                predictions.append(model.predict(observation, action))
            observation = np.mean(predictions, axis=0)

            self.future_state_buffer.append(observation)

        return observation

    def learn(self):
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        obs = samples[:, 0]
        obs_ = samples[:, 1]

        obs = np.array([item for item in obs])
        obs_ = np.array([item for item in obs_])

        for model in self.dc_models:
            model.train()
            model.learn(obs, obs_)
            model.eval()

    def recalibrate(self, observation):
        if self.future_state_buffer:
            predicted_state = self.future_state_buffer.popleft()
        else:
            raise Exception("Delay must be greater than 0.")

        difference = observation - predicted_state

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            x[i] += difference

        self.future_state_buffer = deque(x)
        self.future_state += difference

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
