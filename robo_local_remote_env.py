import gym
from gym import error, utils
from gym.utils import seeding
from gym.envs.registration import register
from gym.spaces import Dict, Box

import numpy as np
import math
import os

from stable_baselines3 import SAC
from collections import OrderedDict, namedtuple

import sys
import random
import torch
import utils
import json

import sys
sys.path.insert(0, '/home/server00/Luc/git_delay_arm_sim_2/predictive')
from undelay_wrapper import UnDelayWrapper#, EMPTYUnDelayWrapper
from wrappers_rd import UnseenRandomDelayWrapper
import pickle

def norm_obs(obs):
    # np.array(list(obs.values())).flatten()
    return np.concatenate([x.reshape(-1) for x in obs.values()]).astype(np.float32)

# LEnv is the normal environment

# Remote Environment (Remote)
class REnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, env_id="FetchReach-v1", seed=None):
        super(REnv, self).__init__()
        # Make the values adaptive to environmennt
        self.remote_Env = gym.make(env_id)
        self.operator_Env = gym.make(env_id)
        seed = np.random.randint(0,100) if not seed else seed

        # print("remote",seed)
        model_dir = json.load(open("./working_models.json", "r"))[env_id]
        self.operator = SAC.load(model_dir, env=self.remote_Env)
    
        self.remote_Env.seed(seed)
        self.operator_Env.seed(seed)

        self.action_space = self.remote_Env.action_space
        self.observation_space = self.remote_Env.observation_space
        # obs_shape = len(norm_obs(self.time_step.observation))
        # self.observation_space = Box(low=self.remote_env.observation_space.low, high=self.remote_env.observation_space.high, shape=(obs_shape,), dtype=np.float32)

        self.remote_observation = self.remote_Env.reset()
        self.operator_observation = self.operator_Env.reset()

        self.observation_space['observation'] = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.observation_space['achieved_goal'] = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space['desired_goal'] = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)


    def step(self, action): # Make sure the operator obs is a copy
        self.remote_observation, remote_reward, remote_done, remote_info = self.remote_Env.step(action)

        operator_action, _states = self.operator.predict(self.operator_observation, deterministic=True)
        self.operator_observation, operator_reward, operator_done, operator_info = self.operator_Env.step(operator_action)

        # Modify obs
        self.remote_observation = format_obs(self.remote_observation, self.operator_observation)

        # Euclidean distance
        reward = -np.linalg.norm(self.remote_observation['achieved_goal'] - self.remote_observation['desired_goal'])

        reward = max(-2, reward) # Clip reward
        return self.remote_observation, float(reward), remote_done, {}


    def reset(self):
        self.remote_observation = self.remote_Env.reset()
        self.remote_observation = self.operator_Env.reset()

        return format_obs(self.remote_observation, self.operator_observation)

    def render(self, mode='human'):
        return self.remote_Env.render()

    def close(self):
        self.operator_Env.close()
        self.remote_Env.close()
    
    # HER
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = float(-np.linalg.norm(achieved_goal - desired_goal))
        return reward
        


'''
obs (25,):
                grip_pos, [0, 1, 2]
                object_pos [3, 4, 5]],
                object_rel_pos [6, 7, 8],
                gripper_state [9, 10, 11] # robot qpos[-2:] not sure but doesn't seem related to object I think it's the change in velocity or torque as when the action is 0 it is always [0,0,-3.85],
                object_rot [12, 13] # as 14 seems to change?,
                object_velp [14, 15, 16] # Seems to be relative velocity of object to gripper ,
                object_velr [17, 18, 19] # rotational velocity of object,
                grip_velp [20,21,22] # velocity of gripper,
                gripper_vel [23,24,25],

Need to replace object with target
get rid of achieved goal and desired goal stuff
''' 

# Remote Environment (Remote)
class REnvPD(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, env_id="FetchPush-v1", seed=None):
        super(REnvPD, self).__init__()
        # Make the values adaptive to environmennt
        self.remote_Env = gym.make(env_id)
        self.operator_Env = gym.make(env_id)
        seed = np.random.randint(0,100) if not seed else seed

        model_dir = json.load(open("./working_models.json", "r"))[env_id]
        self.operator = SAC.load(model_dir, env=self.remote_Env)
        self.remote_Env.seed(seed)
        self.operator_Env.seed(seed)

        self.prev_error = [0,0,0]

        # self.action_space = spaces.Box(low=-14, high=14, shape=(2,), dtype=np.float32)
        #max = int(self.Env.action_spec().maximum[0]) #* 14
        #min = int(self.Env.action_spec().minimum[0]) #* 14

        #self.action_space = spaces.Box(low=min, high=max, shape=(2,), dtype=np.float32)
        #obs_shape = len(norm_obs(self.time_step.observation))

        #self.observation_space = #Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        self.observation_space = self.remote_Env.observation_space
        self.observation_space['observation'] = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.observation_space['achieved_goal'] = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.observation_space['desired_goal'] = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)


        self.action_space = Box(low=-80, high=80, shape=(2,))


       # assert np.array_equal(self.operator_time_step.observation['to_target'], self.time_step.observation['to_target'])


        # Modified for remote
    def step(self, action, obs): # Make sure the operator obs is a copy
        # PD Controller
        P, D = (action[0], action[1]) # for position
        #g_P, g_D = (action[2], action[3]) # For gripper
        #error = self.time_step.observation['position'] - self.time_step.observation['to_target']
        # Only works for reacher so maybe changes this
        # print(obs)
        error = obs["achieved_goal"] - obs["desired_goal"]
        #error = np.append(error[:3])#, sum(error[3:]))
        proportional = error
        derivative = error - self.prev_error

        pd_value = P * proportional + D * derivative
        self.r_obs, r_rew, r_done, r_info = self.remote_Env.step(np.append(pd_value, [0]))
        self.prev_error = error

        #operator_obs = norm_obs(self.operator_time_step.observation)
        operator_action, _states = self.operator.predict(self.o_obs, deterministic=True)
        self.o_obs, o_rew, o_done, o_info = self.operator_Env.step(operator_action)

        # Modify obs
        # self.time_step.observation['to_target'] = self.operator_time_step.observation['position']
        self.r_obs = format_obs(self.r_obs, self.o_obs)


        # Euclidean distance
        reward = -np.linalg.norm(self.r_obs['achieved_goal'] - self.r_obs['desired_goal'])
        # Sigmoid normalisation
        # reward = 1 / (1 + math.exp(reward/100))
        #reward = -utils.softmax([0,reward,1])[1] # Normalise within range
        # reward = max(-2, reward) # Helps but not not for PD?
        return self.r_obs, float(reward), r_done, {"operator_reward": o_rew}


    def reset(self):

        self.r_obs = self.remote_Env.reset()
        self.o_obs = self.operator_Env.reset()

        return format_obs(self.r_obs, self.o_obs) # reward, done, info can't be included

    def render(self, mode='human'):
        return self.remote_Env.render()

    def close(self):
        self.operator_Env.close()
        self.remote_Env.close()
    
    # HER
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = float(-np.linalg.norm(achieved_goal - desired_goal))
        return max(-1,reward)

# Remote Environment (Remote)
class REnvPDNormObs(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):
        super(REnvPDNormObs, self).__init__()
        #print("REnvPDNormObs!")

        # Make the values adaptive to environmennt
        self.remote_Env = gym.make(env_id)
        self.operator_Env = gym.make(env_id)
        seed = np.random.randint(0,100) if not seed else seed
        seed = 0
        self.fname = fname
        #self.n = max(len(os.listdir("dcac_scores"))-1, 0) # Only update the latest

        model_dir = json.load(open("./working_models.json", "r"))[env_id]
        self.operator = SAC.load(model_dir, env=self.remote_Env)
        self.remote_Env.seed(seed)
        self.operator_Env.seed(seed)

        self.prev_error = [0,0,0]

        # self.action_space = spaces.Box(low=-14, high=14, shape=(2,), dtype=np.float32)
        #max = int(self.Env.action_spec().maximum[0]) #* 14
        #min = int(self.Env.action_spec().minimum[0]) #* 14

        #self.action_space = spaces.Box(low=min, high=max, shape=(2,), dtype=np.float32)
        #obs_shape = len(norm_obs(self.time_step.observation))

        #self.observation_space = #Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)
        # self.observation_space = self.remote_Env.observation_space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        #self.observation_space['achieved_goal'] = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        #self.observation_space['desired_goal'] = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)


        self.action_space = Box(low=-80, high=80, shape=(2,))
        self.rew_count = 0
        self.reward = 0
        # pickle.dump(self, open("./env.pickle", "wb"))
        # return self
       # assert np.array_equal(self.operator_time_step.observation['to_target'], self.time_step.observation['to_target'])

    def get_self(self):
        #self.__init__()
        return self

        # Modified for remote
    def step(self, action): # Make sure the operator obs is a copy
        # PD Controller
        P, D = (action[0], action[1]) # for position
        error = self.r_obs['achieved_goal'] - self.r_obs['desired_goal']
        #error = np.append(error[:3])#, sum(error[3:]))
        proportional = error
        derivative = error - self.prev_error

        pd_value = P * proportional + D * derivative
        self.r_obs, r_rew, r_done, r_info = self.remote_Env.step(np.append(pd_value, [0]))
        self.prev_error = error

        #operator_obs = norm_obs(self.operator_time_step.observation)
        operator_action, _states = self.operator.predict(self.o_obs, deterministic=True)
        self.o_obs, o_rew, o_done, o_info = self.operator_Env.step(operator_action)

        # Modify obs
        # self.time_step.observation['to_target'] = self.operator_time_step.observation['position']
        self.r_obs = format_obs(self.r_obs, self.o_obs)


        # Euclidean distance
        reward = -np.linalg.norm(self.r_obs['achieved_goal'] - self.r_obs['desired_goal'])
        self.reward = reward
        #reward_v = -np.linalg.norm(self.r_obs['observation'][[-5,-4,-3]])
        #reward = reward_p + reward_v
        # Sigmoid normalisation
        # reward = 1 / (1 + math.exp(reward/100))
        #reward = -utils.softmax([0,reward,1])[1] # Normalise within range
        # reward = max(-2, reward) # Helps but not not for PD?
        self.rew_count += float(reward)
        return self.r_obs['observation'], float(reward), r_done, {"operator_reward": o_rew}


    def reset(self):
        #if not self.fname:
            #fname = f"./dcac_scores/DCAC_scores{self.n}.csv"
            
        #with open(fname,"a") as f:
            #f.write(str(self.rew_count)+",")
        self.rew_count = 0 # remember to get rid of 0's at the end
        self.r_obs = self.remote_Env.reset()
        self.o_obs = self.operator_Env.reset()

        return format_obs(self.r_obs, self.o_obs)['observation'] # reward, done, info can't be included

    def render(self, mode='human'):
        return self.remote_Env.render()

    def close(self):
        self.operator_Env.close()
        self.remote_Env.close()
    
    # HER
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = float(-np.linalg.norm(achieved_goal - desired_goal))
        return max(-1,reward)


# For both REnv's PUSH ENV
def format_obs(r_obs, o_obs):

    def delete_idxs(a, idxs):
        for i in idxs:
            a = np.append(a[:i],a[i+1:])
        return a
    
    # Remove the object info from observation
    r_obs['observation'] = delete_idxs(r_obs['observation'], [17,18,19]) # object velr
    r_obs['observation'] = delete_idxs(r_obs['observation'], [14,15,16]) # object velp
    r_obs['observation'] = delete_idxs(r_obs['observation'], [12,13]) # object rot
    r_obs['observation'] = delete_idxs(r_obs['observation'], [6,7,8]) # object rel pos
    r_obs['observation'] = delete_idxs(r_obs['observation'], [3,4,5]) # object pos position 
    # Enter new information about operator
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[0,1,2]]) # Operator position
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[9,10,11]]) # Operator gripper state
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[20,21]]) # Operator gripper velocity
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[22,23,24]]) # Operator gripper velocity

    # r_obs['desired_goal'] = np.append(o_obs['observation'][[0,1,2]])# o_obs['observation'][[9, 10, 11]]) # Operator position and gripper state
    # r_obs['achieved_goal'] = np.append(r_obs['observation'][[0,1,2]])#, r_obs['observation'][[3,4,5]]) # Remote position
    r_obs['achieved_goal'] = r_obs['observation'][[0,1,2]]#, r_obs['observation'][[3,4,5]]) # Remote
    r_obs['desired_goal'] = o_obs['observation'][[0,1,2]]# o_obs['observation'][[9, 10, 11]]) # Operator position and gripper state
    #r_obs['achieved_goal'] = r_obs['observation'][[0,1,2]]#, r_obs['observation'][[3,4,5]]) # Remote position

    for key in r_obs.keys():
        r_obs[key] = r_obs[key].astype(np.float32)

    return r_obs
'''
obs (25,):
    grip_pos, [0, 1, 2]
    object_pos [3, 4, 5]],
    object_rel_pos [6, 7, 8],
    gripper_state [9, 10, 11] # robot qpos[-2:] not sure but doesn't seem related to object I think it's the change in velocity or torque as when the action is 0 it is always [0,0,-3.85],
    object_rot [12, 13] # as 14 seems to change?,
    object_velp [14, 15, 16] # Seems to be relative velocity of object to gripper ,
    object_velr [17, 18, 19] # rotational velocity of object,
    grip_velp [20,21,22] # velocity of gripper,
    gripper_vel [23,24,25],

Things that don't move: (array([ 9, 10, 11, 12, 13, 17, 18, 19, 23, 24]),) remember 25 doesnt exist for some reason


Need to replace object with target
get rid of achieved goal and desired goal stuff REnvPDNormObs

        # super(REnvUndelayed_8, self).__init__(env=env, delay=8, act_delay_range=act_delay_range, env_id=env_id, pretrain=False, n_models=5)
        # print("Undelayed_8!")

        # env = super()#.get_self()

        # self.env = UnDelayWrapper(env, delay=8, env_id="FetchPush-RemotePDNorm-v0", pretrain=False, n_models=5)
        
        # self.reset = self.env.reset
        # self.step = self.env.step

        
        # print(env)
        # print(env.reset())
        # print(self)
        # print(self.reset())


        # env_id = "FetchPush-RemotePDNorm-v0"
        # env = gym.make(env_id)
        # self.observation_space = env.observation_space
        # self.action_space = env.action_space
        # self.env

        # print("OBS SPACE:", self.observation_space)
        # UnDelayWrapper(env, delay=8, env_id=env_id, pretrain=True, n_models=5)
'''
# class REnvUndelayed4_4(UnDelayWrapper(env=REnvPDNormObs(), delay=8, env_id="FetchPush-RemotePDNorm-v0", pretrain=True, n_models=5)):
# Try Inheritance in different order
class REnvUndelayed_8(gym.Env):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):
        super(REnvUndelayed_8, self).__init__()
        env_id = "FetchPush-RemotePDNorm-v0"
        env = gym.make(env_id)
        self.env = UnDelayWrapper( UnseenRandomDelayWrapper (gym.make(env_id),
                obs_delay_range=range(0, 1),
                act_delay_range=range(8, 9)),
                delay=8,
                env_id=env_id,
                pretrain=False,
                n_models=5)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class REnvUndelayed_16(gym.Env):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):
        super(REnvUndelayed_16, self).__init__()

        env_id = "FetchPush-RemotePDNorm-v0"
        env = gym.make(env_id)
        self.env = UnDelayWrapper( UnseenRandomDelayWrapper (gym.make(env_id),
                obs_delay_range=range(0, 1),
                act_delay_range=range(20, 21)),
                delay=10,
                env_id=env_id,
                pretrain=False,
                n_models=5)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class REnvUndelayed_24(gym.Env):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):
        super(REnvUndelayed_24, self).__init__()

        env_id = "FetchPush-RemotePDNorm-v0"
        env = gym.make(env_id)
        self.env = UnDelayWrapper( UnseenRandomDelayWrapper (gym.make(env_id, seed=seed),
                obs_delay_range=range(0, 1),
                act_delay_range=range(24, 25)),
                delay=24,
                env_id=env_id,
                pretrain=False,
                n_models=5)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

'''
class REnvUndelayed_16(UnseenRandomDelayWrapper, UnDelayWrapper, REnvPDNormObs):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):
        env_id = "FetchPush-RemotePDNorm-v0"
        env = gym.make(env_id)
        obs_delay_range=range(0, 1) # Add 0 obs delay
        act_delay_range=range(16, 17) # Add 8 act delay

        super(REnvUndelayed_16, self).__init__(env=env, delay=16, act_delay_range=act_delay_range, env_id=env_id, pretrain=False, n_models=5)

class REnvUndelayed_24(REnvPDNormObs, UnDelayWrapper, UnseenRandomDelayWrapper):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):

        env = gym.make("FetchPush-RemotePDNorm-v0")
        obs_delay_range=range(0, 1) # Add 0 obs delay
        act_delay_range=range(24, 25) # Add 8 act delay

        super(REnvUndelayed_24, self).__init__(env=env, delay=24, act_delay_range=act_delay_range, env_id="FetchPush-RemotePDNorm-v0", pretrain=False, n_models=5)

class REnvUndelayed_32(UnDelayWrapper, UnseenRandomDelayWrapper, REnvPDNormObs):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):

        env = gym.make("FetchPush-RemotePDNorm-v0")
        obs_delay_range=range(0, 1) # Add 0 obs delay
        act_delay_range=range(32, 33) # Add 8 act delay

        super(REnvUndelayed_32, self).__init__(env=env, delay=32, act_delay_range=act_delay_range, env_id="FetchPush-RemotePDNorm-v0", pretrain=False, n_models=5)
'''

'''
# Remote Environment (Remote)
class REnvNormalObsPDOLD(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, env_id="FetchReach-v1", seed=None):
        super(REnvNormalObsPD, self).__init__()
        # Make the values adaptive to environmennt
        self.remote_Env = gym.make(env_id)
        self.operator_Env = gym.make(env_id)
        seed = np.random.randint(0,100) if not seed else seed

        # print("remote",seed)
        model_dir = json.load(open("./working_models.json", "r"))[env_id]
        self.operator = SAC.load(model_dir, env=self.remote_Env)
    
        self.remote_Env.seed(seed)
        self.operator_Env.seed(seed)

        self.action_space = self.remote_Env.action_space
        self.remote_observation = self.remote_Env.reset()
        self.operator_observation = self.operator_Env.reset()
        # Feor determining PD error
        # self.error_range = self.remote_Env.observation_space["achieved_goal"].shape[0] +\
        #                    self.remote_Env.observation_space["desired_goal"].shape[0] 

        #obs_shape = len(self.remote_observation)
        # print(obs_shape, self.remote_Env.observation_space['observation'].low)
        #print("LEN",self.remote_Env.observation_space['observation'].shape[])
        #obs_shape = len(self.remote_Env.observation_space['observation']) + len(self.remote_Env.observation_space['achieved_goal']) + len(self.remote_Env.observation_space['desired_goal'])
        obs_shape = self.remote_Env.observation_space['observation'].shape[0] + self.remote_Env.observation_space['achieved_goal'].shape[0] + self.remote_Env.observation_space['desired_goal'].shape[0]

        #self.observation_space = Box(low=[-np.inf]*obs_shape, high=[np.inf]*obs_shape, shape=(obs_shape,), dtype=np.float32)
        self.observation_space = Box(low=np.array(-np.inf)*np.ones(obs_shape), high=np.array(np.inf)*np.ones(obs_shape), shape=(obs_shape,), dtype=np.float32)


    def step(self, action, obs): # Make sure the operator obs is a copy
        P, D = (action[0], action[1])

        #error = self.time_step.observation['position'] - self.time_step.observation['to_target']
        # Only works for reacher so maybe changes this
        # print(obs)
        error = obs[3,4,5] - obs[0,1,2] # pos - tar_pos

        proportional = error
        derivative = error - self.prev_error

        pd_value = P * proportional + D * derivative

        self.remote_observation, remote_reward, remote_done, remote_info = self.remote_Env.step(action)
        self.prev_error = error
        ##
        # operator_obs = norm_obs(self.operator_observation)
        operator_action, _states = self.operator.predict(self.operator_observation, deterministic=True)
        operator_obs, operator_reward, operator_done, operator_info = self.operator_Env.step(operator_action)
        self.operator_observation = operator_obs
        
        # Modify obs
        self.remote_observation['desired_goal'] = self.operator_observation['achieved_goal']

        # Euclidean distance
        reward = -np.linalg.norm(self.remote_observation['achieved_goal'] - self.remote_observation['desired_goal'])

        reward = max(-2, reward) # Clip reward
        normed_remote_observation = norm_obs(self.remote_observation)
        return normed_remote_observation, float(reward), remote_done, {}


    def reset(self):
        self.prev_error = None # edit
        self.remote_observation = self.remote_Env.reset()
        self.remote_observation = self.operator_Env.reset()
        self.remote_observation['desired_goal'] = self.operator_observation['achieved_goal']
        normed_remote_observation = norm_obs(self.remote_observation)
        return normed_remote_observation # reward, done, info can't be included

    # Takes flat obs as input
    def rectify_obs(obs):
        obs[3,4,5] = self.operator_observation['observation'][0,1,2] # obj_pos -> target_pos
        obs[6,7,8] = self.operator_observation['observation'][0,1,2] - obs[0,1,2] # obj_rel_pos -> target_rel_pos
        obs[14,15,16] = self.operator_observation['observation'][23,24,25]- obs[23,24,25] # obj_velp -> target_velp

        del obs[12,13] # delete object rotation from state info (after as to not mess up indexes)
        del obs[17,18,19]
        del obs[20,21,22] # Not sure if this is relative to gripper or not. Investigate further

    def render(self, mode='human'):
        return self.remote_Env.render()

    def close(self):
        self.operator_Env.close()
        self.remote_Env.close()


## TEST ENV

class TestEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, env_id=("reacher", "hard"), seed=None):
        super(LEnv, self).__init__()

        self.Env = suite.load(*env_id)

        if seed:
            self.Env.task.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) # probably only need the first line here

        self.time_step = self.Env.reset()

        self.env_id = env_id

        self.action_space = spaces.Box(low=min, high=max, shape=self.Env.action_spec().shape, dtype=np.float32)


        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)


    def step(self, action):

        return action, action, self.time_step.last(), {}


    def reset(self):

        return 0 # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        self.Env.close()


# Remote Environment (Remote)
class REnvNormalObs(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, env_id="FetchReach-v1", seed=None):
        super(REnvNormalObs, self).__init__()
        # Make the values adaptive to environmennt
        self.remote_Env = gym.make(env_id)
        self.operator_Env = gym.make(env_id)
        seed = np.random.randint(0,100) if not seed else seed

        # print("remote",seed)
        model_dir = json.load(open("./working_models.json", "r"))[env_id]
        self.operator = SAC.load(model_dir, env=self.remote_Env)
    
        self.remote_Env.seed(seed)
        self.operator_Env.seed(seed)

        self.action_space = self.remote_Env.action_space
        self.remote_observation = self.remote_Env.reset()
        self.operator_observation = self.operator_Env.reset()

        #obs_shape = len(self.remote_observation)
        # print(obs_shape, self.remote_Env.observation_space['observation'].low)
        #print("LEN",self.remote_Env.observation_space['observation'].shape[])
        #obs_shape = len(self.remote_Env.observation_space['observation']) + len(self.remote_Env.observation_space['achieved_goal']) + len(self.remote_Env.observation_space['desired_goal'])
        obs_shape = self.remote_Env.observation_space['observation'].shape[0] + self.remote_Env.observation_space['achieved_goal'].shape[0] + self.remote_Env.observation_space['desired_goal'].shape[0]

        #self.observation_space = Box(low=[-np.inf]*obs_shape, high=[np.inf]*obs_shape, shape=(obs_shape,), dtype=np.float32)
        self.observation_space = Box(low=np.array(-np.inf)*np.ones(obs_shape), high=np.array(np.inf)*np.ones(obs_shape), shape=(obs_shape,), dtype=np.float32)


    def step(self, action): # Make sure the operator obs is a copy
        self.remote_observation, remote_reward, remote_done, remote_info = self.remote_Env.step(action)
        #operator_obs = norm_obs(self.operator_observation)
        operator_action, _states = self.operator.predict(self.operator_observation, deterministic=True)
        operator_obs, operator_reward, operator_done, operator_info = self.operator_Env.step(operator_action)
        self.operator_observation = operator_obs
        
        # Modify obs
        self.remote_observation['desired_goal'] = self.operator_observation['achieved_goal']

        # Euclidean distance
        reward = -np.linalg.norm(self.remote_observation['achieved_goal'] - self.remote_observation['desired_goal'])*20

        reward = max(-2, reward) # Clip reward
        normed_remote_observation = norm_obs(self.remote_observation)
        return normed_remote_observation, float(reward), remote_done, {}


    def reset(self):
        self.remote_observation = self.remote_Env.reset()
        self.remote_observation = self.operator_Env.reset()
        self.remote_observation['desired_goal'] = self.operator_observation['achieved_goal']
        normed_remote_observation = norm_obs(self.remote_observation)
        return normed_remote_observation # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        self.operator_Env.close()
        self.remote_Env.close()
# register(
#     id="Custom-Reacher-Local-v0",
#     entry_point="gym.envs.classic_control:LEnv",
#     max_episode_steps=2000,
# )
#
# register(
#     id="Custom-Reacher-Remote-v0",
#     entry_point="gym.envs.classic_control:REnv",
#     max_episode_steps=2000,
# )
'''
'''
class REnvUndelayed4_4(REnvPDNormObs):
    def __init__(self, env_id="FetchPush-v1", seed=None, fname=None):
        super().__init__()

        seed = np.random.randint(0,100) if not seed else seed
        OBS_DELAY = 4
        ACT_DELAY = 3 # Because of rlrd
        PRETRAIN = False
        N_MODELS = 5
        # Undelay environments
        self.remote_Env = UnDelayWrapper(self.remote_Env, delay=ACT_DELAY+1, env_id="FetchPush-undelayR-v0", pretrain=PRETRAIN, n_models=N_MODELS) # Action delays effect remote

        self.operator_Env = UnDelayWrapper(self.operator_Env, delay=OBS_DELAY, env_id="FetchPush-undelayL-v0", pretrain=PRETRAIN, n_models=N_MODELS) # Obs delays effect operator

        self.remote_Env.seed(seed)
        self.operator_Env.seed(seed)
        # Why isn't the obs being normalised?

class LREnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, env_id, seed=None):
        super(LREnv, self).__init__()
        # Make the values adaptive to environmennt
        self.action_space = spaces.Box(low=-14, high=14, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

        self.Env = suite.load(*env_id)
        self.operator_Env = suite.load(*env_id)
        #
        if not seed:
            seed = np.random.randint(0,100)

        # print("remote",seed)
        self.operator = SAC.load("sac_operator")

        self.Env.task.random.seed(seed)
        self.operator_Env.task.random.seed(seed)

        self.time_step = self.Env.reset()
        self.operator_time_step = self.operator_Env.reset()

        # print(np.array_equal(self.operator_time_step.observation.values, self.time_step.observation.values))
        # print((self.operator_time_step.observation.values(), self.time_step.observation.values()))

        assert np.array_equal(self.operator_time_step.observation['to_target'], self.time_step.observation['to_target'])


        # Modified for remote
    def step(self, action): # Make sure the operator obs is a copy
        self.time_step = self.Env.step(action)

        operator_obs = self.norm_obs(self.operator_time_step.observation)
        operator_action, _states = self.operator.predict(operator_obs, deterministic=True)
        self.operator_time_step = self.operator_Env.step(operator_action)

        # Modify obs
        self.time_step.observation['to_target'] = self.operator_time_step.observation['position']

        # Euclidean distance
        reward = -np.linalg.norm(self.time_step.observation['position'] - self.time_step.observation['to_target'])
        # Sigmoid normalisation
        # reward = 1 / (1 + math.exp(reward/100))
        return self.norm_obs(self.time_step.observation), float(reward/100), self.time_step.last(), {}


    def reset(self):
        self.time_step = self.Env.reset()
        self.operator_time_step = self.operator_Env.reset()

        return self.norm_obs(self.time_step.observation) # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def norm_obs(self, obs):
        return np.array(list(obs.values())).flatten().astype(np.float32)

    def close(self):
        self.Env.close()'''
