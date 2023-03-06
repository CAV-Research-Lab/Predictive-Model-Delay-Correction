import gym
import torch
import torch.nn.functional as F
from delay_correcting_nn import DCNN, run
import pickle
from collections import deque
import numpy as np
import os
import atexit
import sys
import pandas as pd

# VARIANCES = []

from csv import writer


class PMDC(gym.Wrapper):
    def __init__(self, env, delay, env_id, pretrain, n_models, act_delay_range=None):
        super().__init__(env)
        super()
        self.observation_space = env.observation_space
        self.env = env
        self.delay = delay
        self.env_id = env_id
        self.pretrain = pretrain
        self.n_models = n_models

        self.future_state = None
        self.future_state_buffer = deque() # Also contains actions that led to those states

        self.buffer_size = 10_000 # was 5_000
        self.batch_size =  256
        self.start_training = 1000
        self.replay_buffer = deque(maxlen=self.buffer_size)

        ##
        self.layer_size = 128
        self.n_layers = 2

        ##
        self.d_reward = 0
        self.d_reward_total = 0
        self.first = False # for callback
        self.log = pd.DataFrame({"Step":[],"Delayed episodic reward":[]})

        params = pickle.load(open(f"./models/FetchPush-RemotePDNorm-v0/1_step_params.pickle", "rb")) # Using the same params for now but could use env_id
        self.dc_models = [DCNN(**params).eval() for _ in range(n_models)]
        for model in self.dc_models:
            model.learning_rate = np.random.randint(-100,100)/1_200_000 # 1_000_000
        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])

        # Fixed error for recreating env
        if pretrain:
            for i in range(n_models):
                model_file = f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd_{i}.pt"
  
                if not os.path.isfile(model_file):
                    # print(f"xxx Model {i} File Not Found xxx")
                    if pretrain:
                        print(f"--- Creating Pretrain-Model {i} ---")
                        run(env_id=self.env_id,  epochs=4, batch_size=256, file_name=model_file)


                state_dict = torch.load(model_file)
                self.dc_models[i].eval()
                print(f"--- Loaded Model {i} ---")
                self.dc_models[i].load_state_dict(state_dict)


    def reset(self):
        obs = super().reset()
        self.d_reward_total = 0


        self.prev_obs = obs
        self.future_state_buffer = deque()

        self.future_state = self.initial_undelay(obs)

        # Train undelay model
        if len(self.replay_buffer) > self.start_training:
            self.learn()
        return self.future_state

    def step(self, action):

        observation, d_reward, done, info = super().step(action)

        self.d_reward = d_reward
        self.d_reward_total += d_reward

        training_data = (np.append(self.prev_obs.astype(np.float32), action.astype(np.float32)).astype(np.float32), observation.astype(np.float32))
        self.replay_buffer.append(training_data)

        self.recalibrate(observation)
        
        # Ensemble Average
        predictions = []
        for model in self.dc_models:
            predictions.append(model.predict(self.future_state, action))
        self.future_state = np.mean(predictions, axis=0) # KEEP THIS IN
        # variance = np.mean(np.var(predictions,axis=0))
        # VARIANCES.append(variance)
        # print(f"Predictions: \n{predictions}\n\nMean: \n{self.future_state}\n\nVariance:\n{variance}-----------------")
        # self.future_state = self.dc_model.predict(self.future_state, action) # Most current future state

        self.future_state_buffer.append(self.future_state) # Append future state and action that led to it SHOULD THIS GO BEFORE?

        reward = self.calculate_reward(self.future_state)

        return self.future_state, reward, done, {"Delayed Reward" : self.d_reward_total} # updated info and reward
        
    # Undelays so that model starts at correct point in future
    def initial_undelay(self, observation):
        action = [0]*self.action_space.shape[0]
        for step in range(self.delay):
            predictions = []
            self.prev_obs = observation

            for model in self.dc_models:
                predictions.append(model.predict(observation, action))
            observation = np.mean(predictions, axis=0) # KEEP THIS IN

            self.future_state_buffer.append(observation)

        return observation

    # Update the model
    def learn(self):
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        obs = samples[:,0] # Predicted state and action to arrive at obs
        obs_ = samples[:,1] # Resultant obs

        obs = np.array([item for item in obs])
        obs_ = np.array([item for item in obs_])

        # print("OBS:\n",obs, "\nNEXT OBS:\n", obs_)
        for model in self.dc_models:
            model.train()
            model.learn(obs, obs_)
            model.eval()

    def recalibrate(self, observation):
        if self.future_state_buffer:
            predicted_state = self.future_state_buffer.popleft() # prediction delay steps ago
        else:
            raise Exception("Delay must be greater than 0.")

        difference = observation - predicted_state

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            x[i] += difference

        self.future_state_buffer = deque(x)
        self.future_state += difference

    # Could learn reward function using nn but we know reward function explicitly in this case
    def calculate_reward(self, obs):
        # print("Obs:",obs)
        reward = -np.linalg.norm(obs[[0,1,2]]- obs[[11,12,13]])
        reward = max(-2, reward) # Clip reward
        return float(reward)

    def save_models(self):
        for i in range(self.n_models):
            model_file = f"./models/{self.env_id}/{self.n_layers}-{self.layer_size}-1_step_prediction_sd_{i}.pt"
            if not os.path.isfile(model_file):
                print("Unable to update model!")
                sys.exit()
            else:
                #print(f"Updated model {i}!")
                torch.save(self.dc_models[i].state_dict(), model_file)

# @atexit.register
# def plot_variance():
#     import matplotlib.pyplot as plt
#     plt.style.use('./plot_style.txt')
#     plt.figure(figsize=(5, 5))  # specify the height and width of the figure
#     plt.title("Ensemble Variance", loc="center")
#     plt.ylim(ymin=0,ymax=5)
#     plt.xlim(xmin=0,xmax=10_000)


#     plt.xlabel("Time steps")
#     plt.ylabel("Variance")
#     plt.plot(VARIANCES)
#     plt.show()
