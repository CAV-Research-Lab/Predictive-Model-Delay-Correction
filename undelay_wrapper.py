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
'''
kEEP TRACK OF DELAYED REWARDS IN FILE INSTEAD

'''
from csv import writer

# os.environ["LD_LIBRARY_PATH"] = "/home/server00/.mujoco/mujoco210/bin"

class UnDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay, env_id, pretrain, n_models, act_delay_range=None):
        super().__init__(env)
        # print("UNDELAY")
        self.observation_space = env.observation_space
        self.env = env
        self.delay = delay
        self.env_id = env_id
        self.pretrain = pretrain
        self.n_models = n_models

        self.d_reward = 0

        self.future_state = None
        self.future_state_buffer = deque() # Also contains actions that led to those states

        self.buffer_size = 10_000 # was 5_000
        self.batch_size =  256
        self.start_training = 1000
        self.replay_buffer = deque(maxlen=self.buffer_size)

        ##
        self.layer_size = 128
        self.n_layers = 2
        self.df = pd.DataFrame({"Delayed Reward":[],"Simulated Reward":[]})

        ##

        params = pickle.load(open(f"./models/FetchPush-RemotePDNorm-v0/1_step_params.pickle", "rb")) # Using the same params for now but could use env_id
        # self.dc_model = DCNN(**params) # Changed to just use 1 step
        self.dc_models = [DCNN(**params).eval() for _ in range(n_models)]
        for model in self.dc_models:
            model.learning_rate = np.random.randint(-100,100)/1_200_000 # 1_000_000
        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])


        '''
        For the current setup to work I will probably have to delete the models so that a new pretrain set is made before each run.
        '''

        # Fixed error for recreating env
        # Creates new file if they dont exist and loads models
        for i in range(n_models):
            model_file = f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd_{i}.pt"

            if not os.path.isfile(model_file):
                if pretrain:
                    print(f"--- Creating Pretrain-Model {i} ---")
                    run(env_id=self.env_id,  epochs=4, batch_size=256, file_name=model_file)
                else:
                    print(f"--- Creating Non-Pretrain-Model {i} ---")
                    run(env_id=self.env_id,  epochs=0, batch_size=10, file_name=model_file)
                    print("xxx Not using Pre-trained Model xxx")


            state_dict = torch.load(model_file)
            self.dc_models[i].load_state_dict(state_dict)
            print(f"--- Loaded Model {i} ---")


    def reset(self):
        obs = super().reset()
        self.prev_obs = obs
        self.future_state_buffer = deque()
        self.d_reward=0
        self.future_state = self.initial_undelay(obs)

        # Train undelay model
        if len(self.replay_buffer) > self.start_training:
            self.learn()
            
        return self.future_state

    def step(self, action):
        observation, d_reward, done, info = super().step(action)

        training_data = (np.append(self.prev_obs.astype(np.float32), action.astype(np.float32)).astype(np.float32), observation.astype(np.float32))
        self.replay_buffer.append(training_data)

        self.recalibrate(observation)
        
        # Ensemble Average
        predictions = []
        for model in self.dc_models:
            predictions.append(model.predict(self.future_state, action))
        # self.future_state = np.mean(predictions, axis=0) # KEEP THIS IN
        # variance = np.mean(np.var(predictions,axis=0))
        # VARIANCES.append(variance)
        # print(f"Predictions: \n{predictions}\n\nMean: \n{self.future_state}\n\nVariance:\n{variance}-----------------")
        # self.future_state = self.dc_model.predict(self.future_state, action) # Most current future state

        self.future_state_buffer.append(self.future_state) # Append future state and action that led to it SHOULD THIS GO BEFORE?
        # Should be previous action
        self.d_reward = d_reward#self.calculate_reward(observation)
        reward = self.calculate_reward(self.future_state)

        df2 = pd.DataFrame({"Delayed Reward":[d_reward],"Simulated Reward":[reward]})
        self.df = pd.concat([self.df, df2], axis=0)

        if done:

            self.df.to_csv(f"./Scores/undelay_scores/sac_scores{len(os.listdir('./Scores/undelay_scores'))}.csv")
            self.df = pd.DataFrame({"Delayed Reward":[],"Simulated Reward":[]})
            self.save_models()
        
        # info["delayed reward"] = d_reward

        return self.future_state, reward, done, info # updated info and reward

    '''
    @atexit.register
    def plot_variance():
        import matplotlib.pyplot as plt
        plt.plot(VARIANCES)
        plt.show()
    '''
        
    # Undelays so that model starts at correct point in future
    def initial_undelay(self, observation):
        action = [0]*self.action_space.shape[0]

        for step in range(self.delay):
            predictions = []
            self.prev_obs = observation

            for model in self.dc_models:
                predictions.append(model.predict(observation, action))
            observation = np.mean(predictions, axis=0) # KEEP THIS IN            self.future_state_buffer.append(observation)

        return observation

    # Update the model
    def learn(self):
        #samples = np.random.choice(self.replay_buffer ,self.batch_size)
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        #observation = np.append(observation, predicted_state[1]) # observation + action that led to predicted state
        #predicted_state = predicted_state[0]
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
        else: # If delay is 0!
            predicted_state = observation

        # predicted state is obs act del steps ago so obs and pred are the same time. To learn we need the obs before so that osb+act = new_obs without predictions at all
        # self.replay_buffer.append([predicted_state[0], np.append(observation, predicted_state[1])]) # prediction, label

        difference = observation - predicted_state

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            x[i] += difference

        self.future_state_buffer = deque(x)
        self.future_state += difference

    # Could learn reward function using nn but we know rf explicitly in this case
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

class OLDUnDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay, env_id, pretrain, n_models, act_delay_range=None):
        super().__init__(env)
        super().seed(0)
        # print("UNDELAY")
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
        # self.dc_model = DCNN(**params) # Changed to just use 1 step
        self.dc_models = [DCNN(**params).eval() for _ in range(n_models)]
        for model in self.dc_models:
            model.learning_rate = np.random.randint(-100,100)/1_200_000 # 1_000_000
        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])


        '''
        For the current setup to work I will probably have to delete the models so that a new pretrain set is made before each run.
        '''

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

        # self.dc_model.eval()

    def reset(self):
        obs = super().reset()
        # self.first = False # for callback
        self.d_reward_total = 0


        self.prev_obs = obs
        self.future_state_buffer = deque()

        self.future_state = self.initial_undelay(obs)

        # Train undelay model
        if len(self.replay_buffer) > self.start_training:
            self.learn()
        return self.future_state

    def step(self, action):
        # if not self.first:
        #     self.d_reward_total = 0
        #     self.d_reward = 0
        #     self.first = True
        #     self.log_values()

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
        # Should be previous action
        # d_reward = self.calculate_reward(observation)
        reward = self.calculate_reward(self.future_state)

        # print(reward, self.d_reward)

        return self.future_state, reward, done, {"Delayed Reward" : self.d_reward_total} # updated info and reward

    

    
    # def log_values(self):

        
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
        #samples = np.random.choice(self.replay_buffer ,self.batch_size)
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        #observation = np.append(observation, predicted_state[1]) # observation + action that led to predicted state
        #predicted_state = predicted_state[0]
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
        else: # If delay is 0!
            raise Exception("Delay must be greater than 0.")
            # predicted_state = observation

        # predicted state is obs act del steps ago so obs and pred are the same time. To learn we need the obs before so that osb+act = new_obs without predictions at all
        # self.replay_buffer.append([predicted_state[0], np.append(observation, predicted_state[1])]) # prediction, label

        difference = observation - predicted_state

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            x[i] += difference

        self.future_state_buffer = deque(x)
        self.future_state += difference

    # Could learn reward function using nn but we know rf explicitly in this case
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

'''
class UnDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay, env_id, pretrain, n_models, act_delay_range=None):
        super().__init__(env)
        # print("UNDELAY")
        self.observation_space = env.observation_space
        self.env = env
        self.delay = delay
        self.env_id = env_id
        self.pretrain = pretrain
        self.n_models = n_models

        self.future_state = None
        self.future_state_buffer = deque() # Also contains actions that led to those states

        self.buffer_size = 5_000
        self.batch_size =  256
        self.start_training = 1000
        self.replay_buffer = deque(maxlen=self.buffer_size)

        ##
        self.layer_size = 128
        self.n_layers = 2

        ##

        params = pickle.load(open(f"./models/FetchPush-RemotePDNorm-v0/1_step_params.pickle", "rb")) # Using the same params for now but could use env_id
        self.dc_model = DCNN(**params) # Changed to just use 1 step
        self.dc_models = [DCNN(**params).eval() for _ in range(n_models)]
        for model in self.dc_models:
            model.learning_rate = np.random.randint(-100,100)/1_000_000
        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])




        # Fixed error for recreating env
        for i in range(n_models):
            model_file = f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd_{i}.pt"
            # if not pretrain:
            #     print("Deleting previously saved model")
            #     os.remove(model_file)

            if not os.path.isfile(model_file):
                # print(f"xxx Model {i} File Not Found xxx")
                if pretrain:
                    print(f"--- Creating Pretrain-Model {i} ---")
                    run(env_id=self.env_id,  epochs=4, batch_size=256, file_name=model_file)
                else:
                    print(f"--- Creating Non-Pretrain-Model {i} ---")
                    run(env_id=self.env_id,  epochs=0, batch_size=10, file_name=model_file)
                    print("xxx Not using Pre-trained Model xxx")


            state_dict = torch.load(model_file)
            self.dc_model.load_state_dict(state_dict)
            print(f"--- Loaded Model {i} ---")

        self.dc_model.eval()

    def reset(self):
        obs = super().reset()
        self.prev_obs = obs
        self.future_state_buffer = deque()

        self.future_state = self.initial_undelay(obs)

        # Train undelay model
        if len(self.replay_buffer) > self.start_training:
            self.learn()
        return self.future_state

    def step(self, action):
        observation, reward, done, info = super().step(action)
        
        if not done:
            training_data = (np.append(self.prev_obs.astype(np.float32), action.astype(np.float32)).astype(np.float32), observation.astype(np.float32))
            self.replay_buffer.append(training_data)

        self.recalibrate(observation)
        
        # Ensemble Average
        predictions = []
        for model in self.dc_models:
            predictions.append(model.predict(self.future_state, action))
        self.future_state = np.mean(predictions, axis=0)
        variance = np.mean(np.var(predictions,axis=0))
        # VARIANCES.append(variance)
        # print(f"Predictions: \n{predictions}\n\nMean: \n{self.future_state}\n\nVariance:\n{variance}-----------------")
        # self.future_state = self.dc_model.predict(self.future_state, action) # Most current future state

        self.future_state_buffer.append(self.future_state) # Append future state and action that led to it SHOULD THIS GO BEFORE?
        # Should be previous action
        reward = self.calculate_reward(self.future_state)

        if done:
            self.save_models()

        return self.future_state, reward, done, info # updated info and reward


    @atexit.register
    def plot_variance():
        import matplotlib.pyplot as plt
        plt.plot(VARIANCES)
        plt.show()

    # Undelays so that model starts at correct point in future
    def initial_undelay(self, observation):
        action = [0]*self.action_space.shape[0]
        for step in range(self.delay):
            observation = self.dc_model.predict(observation,action)
            self.future_state_buffer.append(observation)

        return observation

    # Update the model
    def learn(self):
        #samples = np.random.choice(self.replay_buffer ,self.batch_size)
        samples = np.array(self.replay_buffer, dtype="object")[np.random.choice(len(self.replay_buffer), self.batch_size)]
        #observation = np.append(observation, predicted_state[1]) # observation + action that led to predicted state
        #predicted_state = predicted_state[0]
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
        predicted_state = self.future_state_buffer.popleft() # prediction delay steps ago
        # predicted state is obs act del steps ago so obs and pred are the same time. To learn we need the obs before so that osb+act = new_obs without predictions at all
        # self.replay_buffer.append([predicted_state[0], np.append(observation, predicted_state[1])]) # prediction, label

        difference = observation - predicted_state

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            x[i] += difference

        self.future_state_buffer = deque(x)
        self.future_state += difference

    # Could learn reward function using nn but we know rf explicitly in this case
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






        if pretrain:
            for i in range(n_models):
                model_file = f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd_{i}.pt"
                if not os.path.isfile(model_file):
                    print(f"xxx Model {i} File Not Found xxx")
                    print(f"--- Creating Model {i} ---")
                    run(env_id=self.env_id,  epochs=4, batch_size=256, file_name=model_file)

                state_dict = torch.load(model_file)
                self.dc_model.load_state_dict(state_dict)
                print(f"--- Loaded Pre-trained Model {i} ---")

        else:
            print("xxx Not using Pre-trained Model xxx")
# Undelays the environment using the given model
class UnDelayWrapperNoRB(gym.Wrapper):
    def __init__(self, env, delay, env_id, pretrain):
        super().__init__(env)
        # self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        self.future_state = None
        self.future_state_buffer = deque() # Also contains actions that led to those states
        self.delay = delay
        #self.replay_buffer = []

        params = pickle.load(open(f"./models/{env_id}/1_step_params.pickle", "rb"))
        self.dc_model = DCNN(**params) # Changed to just use 1 step

        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])

        if pretrain:
            model_file = f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd.pt"
            if os.path.isfile(model_file):
                state_dict = torch.load(model_file)
                self.dc_model.load_state_dict(state_dict)
                print("--- Loaded Pre-trained Model ---")
            else:
                print("xxx No Pre-trained Model Found! xxx")
        else:
            print("xxx Not using Pre-trained Model xxx")
        self.dc_model.eval()
        # assert isinstance(env.action_space, gym.spaces.Box)

    def reset(self):
        # self.save_models()
        obs = super().reset()
        self.future_state_buffer = deque()

        self.future_state = self.initial_undelay(obs)
        return self.future_state

    def step(self, action):
        observation, reward, done, info = super().step(action) # Apply recalibration here
        self.recalibrate(observation)
        self.future_state = self.dc_model.predict(self.future_state, action)
        self.future_state_buffer.append((self.future_state, action))
        return self.future_state[0], reward, done, info

    # Undelays so that model starts at correct point in future
    def initial_undelay(self, observation):
        action = [0]*self.action_space.shape[0]
        for step in range(self.delay):
            observation = self.dc_model.predict(observation,action)
            self.future_state_buffer.append((observation, action))

        return observation

    # Update the model
    def learn(self, observation, predicted_state):
        # print(observation)
        observation = np.append(observation, predicted_state[1]) # observation + action that led to predicted state
        predicted_state = predicted_state[0]
        # print(observation, predicted_state)
        self.dc_model.train()
        self.dc_model.learn(observation, predicted_state)
        self.dc_model.eval()

    def recalibrate(self, observation):
        predicted_state = self.future_state_buffer.popleft() # prediction delay steps ago
        #self.replay_buffer.append([observation, predicted_state])
        self.learn(observation, predicted_state)

        difference = observation - predicted_state[0]
        # print(f"Before: {self.future_state_buffer} Difference: {difference}")

        # print("-- Recalibrated --")
        # print(f"Predicted: {predicted_state}, Actual: {observation}, difference: {difference}")
        # print(f"Old Future state: {self.future_state}")
        # print(np.array(self.future_state_buffer), difference)

        x = np.array(self.future_state_buffer)
        for i in range(len(x)):
            x[i][0] += difference


        self.future_state_buffer = deque(x)
        # print(f"After: {self.future_state_buffer}")

        self.future_state += difference
        # print(f"New Future state: {self.future_state}")

# Undelays the environment using the given model
class OldUnDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay, env_id):
        super().__init__(env)
        # self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        self.future_state = None
        self.future_state_buffer = deque() # Also contains actions that led to those states
        self.delay = delay

        params = pickle.load(open(f"./models/{env_id}/1_step_params.pickle", "rb"))
        self.dc_model = DCNN(**params) # Changed to just use 1 step

        n_layers = str(params["n_layers"])
        layer_size = str(params["layer_size"])

        state_dict = torch.load(f"./models/{env_id}/{n_layers}-{layer_size}-1_step_prediction_sd.pt")
        self.dc_model.load_state_dict(state_dict)
        self.dc_model.eval()
        # assert isinstance(env.action_space, gym.spaces.Box)

    def reset(self):
        obs = super().reset()
        self.future_state_buffer = deque()

        self.future_state = self.initial_undelay(obs)
        return self.future_state

    def step(self, action):
        observation, reward, done, info = super().step(action) # Apply recalibration here
        self.recalibrate(observation)
        self.future_state = self.dc_model.predict(self.future_state, action)
        self.future_state_buffer.append((self.future_state, action))
        return self.future_state[0], reward, done, info

    # Undelays so that model starts at correct point in future
    def initial_undelay(self, observation):
        action = [0]*self.action_space.shape[0]
        for step in range(self.delay):
            observation = self.dc_model.predict(observation,action)
            self.future_state_buffer.append((observation, action))

        return observation

    # Update the model
    def learn(self, observation, predicted_state):
        # print(observation)
        observation = np.append(observation, predicted_state[1]) # observation + action that led to predicted state
        predicted_state = predicted_state[0]
        # print(observation, predicted_state)
        self.dc_model.train()
        self.dc_model.learn(observation, predicted_state)
        self.dc_model.eval()

    def recalibrate(self, observation):
        predicted_state = self.future_state_buffer.popleft() # prediction delay steps ago
        self.learn(observation, predicted_state)

        difference = observation - predicted_state[0]
        # print("-- Recalibrated --")
        # print(f"Predicted: {predicted_state}, Actual: {observation}, difference: {difference}")
        # print(f"Old Future state: {self.future_state}")
        self.future_state += difference / self.delay
        # print(f"New Future state: {self.future_state}")
'''
'''
# Undelays the environment using the given model
class RandomUnDelayWrapper(gym.Wrapper): # Doesn't work
    def __init__(self, env, env_id):
        super().__init__(env)
        self.future_state = None
        self.future_state_buffer = deque()
        self.dc_model = DCNN(**pickle.load(open(f"./models/{env_id[0]}/1_step_params.pickle", "rb"))) # Changed to just use 1 step
        state_dict = torch.load(f"./models/{env_id[0]}/1_step_prediction_sd.pt")
        self.dc_model.load_state_dict(state_dict)
        self.dc_model.eval()
        # assert isinstance(env.action_space, gym.spaces.Box)

    def reset(self):
        obs = super().reset()
        self.future_state_buffer = deque()

        self.future_state = self.initial_undelay(obs)
        return self.future_state

    def step(self, action):
        # For now assuming the unAugrandomdelay wrapper is used
        observation, reward, done, info = super().step(action) # Apply recalibration here
        self.recalibrate(observation)
        self.future_state = self.dc_model.predict(self.future_state, action)
        self.future_state_buffer.append(self.future_state)
        return self.future_state, reward, done, info

    # Undelays so that model starts at correct point in future
    def undelay(self, observation, delay):
        for step in range(self.delay):
            observation = self.dc_model.predict(observation,[0]*self.action_space.shape[0])
            self.future_state_buffer.append(observation)

        return observation

    def recalibrate(self, observation):
        predicted_state = self.future_state_buffer.popleft() # prediction delay steps ago
        difference = observation - predicted_state
        # print("-- Recalibrated --")
        # print(f"Predicted: {predicted_state}, Actual: {observation}, difference: {difference}")
        # print(f"Old Future state: {self.future_state}")
        self.future_state += difference / self.delay
        # print(f"New Future state: {self.future_state}")

'''

# Undelays the environment using the given model
class OldUnDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay, env_id):
        super().__init__(env)
        # self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
        self.dc_model = DCNN(**pickle.load(open(f"./models/{env_id[0]}/{delay}_step_params.pickle", "rb")))
        state_dict = torch.load(f"./models/{env_id[0]}/{delay}_step_prediction_sd.pt")
        self.dc_model.load_state_dict(state_dict)
        self.dc_model.eval()
        # assert isinstance(env.action_space, gym.spaces.Box)

    def reset(self):
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.dc_model.predict(observation, action), reward, done, info


class NothingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return super().reset()

    def step(self, action):
        return super.step()



# Augments the state with action buffer of given length
# class StateAugmentation_Wrapper(gym.Wrapper):
#     def __init__(self, delay):
#         super().__init__(env)
#         self.action buffer = deque([0]*delay)






# if __name__ == "__main__":
#     env = gym.make("CartPole-v1")
#     env = UnDelayWrapper(env)

#     print(type(env.reset())
#     obs_, rew, done, info = env.step(1)
#     print(obs_)


# # Applies action delays
# class ConstantDelay(gym.Env):
# 	def __init__(self, env_id):
# 		self.Env = gym.make(env_id)
# 		self.action_space = '...'
# 		self.observation_space = '...'

# 	def step(self, action):
# 		return self.Env.step(action)

# 	def reset(self):
# 		return self.Env.reset()

# 	def render(self):
# 		return self.Env.render()

# 	def close(self):
# 		return self.Env.close()

# # Undelays the environment using model
# class UnDelay(gym.Env):
# 	def __init__(self, env_id):
# 		self.env = gym.make(env_id)

# 	def step(self):
# 		return

# 	def reset(self):
# 		return

# 	def render(self):
# 		return

# 	def close(self):
# 		return
