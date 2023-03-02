import gym

from stable_baselines3 import SAC
from wrappers_rd import *
from undelay_wrapper import OLDUnDelayWrapper, UnDelayWrapper
from stable_baselines3.common.callbacks import BaseCallback
# from local_remote_env import LEnv
import random
import sys
import os
'''
# TODO:
- Look at old pc code and see why that one seemed to work and this one doesnt
Can use just the github version

'''
# Wrap env in one step constant delays


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=1, env=None):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Access the environment and log the desired variable
        # env = self.env
        try:
            value = self.env.d_reward
            self.logger.record("delayed reward", value)
            value = self.env.d_reward_total
            # print(value)
            self.logger.record("delayed episodic reward", value)

            # self.env.d_reward = 0
            # self.env.d_reward_total = 0
        except:
            value = self.env.reward
            self.logger.record("delayed reward", value)
            value = self.env.rew_count
            self.logger.record("delayed episodic reward", value)

            # self.env.reward = 0
            # self.env.rew_count = 0

        return True




def train(env_id, undelay=True, steps=1_000_000, seed=None, replay_buffer=True, pretrain=True, n_models=1, label="", ACT_DELAY=None,unseen=False):
    # if pretrain == False:
        # for i in range(n_models):
            # delay_settings["max_act_del"] += delay_value
            # delay_settings["min_act_del"] += delay_value
            # os.system(f"rm ./models/FetchPush-RemotePDNorm-v0/2-128-1_step_prediction_sd_{i}.pt")
            # print(f"[x] Removed 2-128-1_step_prediction_sd_{i}.")
    delay_v = {8:38, 16:54, 24:70}
    seed = seed if seed else random.randint(0,100)
    print("Action Delay", ACT_DELAY)
    if unseen:
        print("--- Unseen ---") # swap to no RB for comparison but should make a setting
        env = gym.make(env_id, seed=seed)
        env = UnseenRandomDelayWrapper(env, obs_delay_range=range(OBS_DELAY,OBS_DELAY+5), act_delay_range=range(ACT_DELAY-1,ACT_DELAY))


    elif undelay:
        if replay_buffer == True:
            print("--- Undelay Aug ---") # swap to no RB for comparison but should make a setting
            env = OLDUnDelayWrapper( UnseenRandomDelayWrapper (gym.make(env_id, seed=seed),
                obs_delay_range=range(0, 1),
                act_delay_range=range(ACT_DELAY-1, ACT_DELAY)), delay=OBS_DELAY+ACT_DELAY, env_id=env_id, pretrain=pretrain, n_models=n_models)
            env = AugmentedRandomDelayWrapper(env, obs_delay_range=range(OBS_DELAY,OBS_DELAY+5), act_delay_range=range(0,1), delay_v=24)
            # env = UnseenRandomDelayWrapper(env, obs_delay_range=range(OBS_DELAY,OBS_DELAY+5), act_delay_range=range(0,1))#, delay_v=24)

        else:
            print("--- Undelay No RB ---") # swap to no RB for comparison but should make a setting
            # env = UnDelayWrapperNoRB( UnseenRandomDelayWrapper (gym.make(env_id, seed=seed),
            #     obs_delay_range=range(OBS_DELAY, OBS_DELAY+1),
            #     act_delay_range=range(ACT_DELAY, ACT_DELAY+1)), delay=OBS_DELAY+ACT_DELAY+1, env_id=env_id, pretrain=pretrain)
    else:
        print("--- Aug Delay ---")
        env = AugmentedRandomDelayWrapper (gym.make(env_id),
               obs_delay_range=range(OBS_DELAY, OBS_DELAY+5),
               act_delay_range=range(ACT_DELAY-1, ACT_DELAY), delay_v=delay_v[ACT_DELAY])

    mode = 'undelay-random' if undelay else 'Aug-random'
    mode = 'unseen' if unseen else mode
    # import pdb
    # pdb.set_trace()
    env.seed(seed)
    mode = 'ud-gym' if env_id == "FetchPush-undelay-v0" else mode
    settings = str(n_models)+ '_pre' if pretrain else str(n_models)
    model = SAC("MlpPolicy",  env, verbose=0, tensorboard_log=f"./pred_logs/{mode}/{settings}/{label}", buffer_size=20_000, device="cuda:3", learning_starts=20_000) # Changed buf from 1M
    model.learn(total_timesteps=steps, log_interval=1, callback=TensorboardCallback(env=env))

    model.save(f"./pred_rl_models/{mode}/{OBS_DELAY+ACT_DELAY}_step_SAC_{env_id[0]}")

# train(env)
if __name__ == "__main__":
    OBS_DELAY = 0
    ACT_DELAY = 24
    episode_info = []
    episode_steps = []
    env_id = "FetchPush-RemotePDNorm-v0"
    # env_id = "FetchPush-undelay-v0"
    steps = 100_000
    seed = 0
    n_models = 5
    undelay = True
    replay_buffer = True
    pretrain = False
    run_all = True
    unseen = False
    # Train with predictive model
    # ACT_DELAY -= 1  # Remember this is always actually one greater due to wrapper
    '''
    TODO:
     - Get Aug-undelay results for each delay type
     - Try out the new Undelay wrapper here to see if saving and loading the models is the issue with DCAC.

    '''
    if not run_all:

        train(env_id=env_id, undelay=undelay, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=pretrain, n_models=n_models, label=ACT_DELAY, ACT_DELAY=ACT_DELAY,unseen=unseen)

    else:
        n_models_test = [1,5,10]

        # for n_models in n_models_test:
        #     act_del = ACT_DELAY

        #     ACT_DELAY = act_del
        #     # Undelay
        #     train(env_id=env_id, undelay=True, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=False, n_models=n_models, label=act_del, ACT_DELAY=act_del,unseen=unseen)
        #     # Aug
        #     train(env_id=env_id, undelay=False, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=pretrain, n_models=n_models, label=act_del, ACT_DELAY=act_del,unseen=False)
        #     # Unseen
        #     train(env_id=env_id, undelay=undelay, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=pretrain, n_models=n_models, label=act_del, ACT_DELAY=act_del,unseen=True)


        ACT_DELAYS = [8,16,24]

        for act_del in ACT_DELAYS:
            seed = act_del

            ACT_DELAY = act_del
            # Undelay
            train(env_id=env_id, undelay=True, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=False, n_models=n_models, label=act_del, ACT_DELAY=act_del,unseen=unseen)
            # Aug
            train(env_id=env_id, undelay=False, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=pretrain, n_models=n_models, label=act_del, ACT_DELAY=act_del,unseen=False)
            # Unseen
            train(env_id=env_id, undelay=undelay, steps=steps, seed=seed, replay_buffer=replay_buffer, pretrain=pretrain, n_models=n_models, label=act_del, ACT_DELAY=act_del,unseen=True)


    # Train without predictive model


    # train(env_id, steps, seed) 88/90 pretrain 89/91 Not-pretrain




    '''
    # Fix this for testing?
    model = SAC.load(f"./pred_rl_models/SAC_{env_id[0]}")
    done = False
    env = LEnv
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()'''
