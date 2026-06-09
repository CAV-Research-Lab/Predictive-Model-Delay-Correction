import gymnasium as gym
import gymnasium_robotics  # noqa: F401 - registers FetchPush-v2 etc.
from gymnasium import error
from gymnasium.envs.registration import register
from gymnasium.spaces import Dict, Box

import numpy as np
import math
import os
import warnings

from stable_baselines3 import SAC
from collections import OrderedDict, namedtuple

import sys
import random
import torch
import json

try:
    import utils as local_utils
except ImportError:
    local_utils = None

try:
    from undelay_wrapper import UnDelayWrapper
except ImportError:
    from PMDC_wrapper import PMDC as UnDelayWrapper
from wrappers_rd import UnseenRandomDelayWrapper
import pickle


FETCH_V2_COMPAT_ENVS = {
    "FetchReach-v2": ("gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv", "sparse"),
    "FetchReachDense-v2": ("gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv", "dense"),
    "FetchPush-v2": ("gymnasium_robotics.envs.fetch.push:MujocoFetchPushEnv", "sparse"),
    "FetchPushDense-v2": ("gymnasium_robotics.envs.fetch.push:MujocoFetchPushEnv", "dense"),
    "FetchSlide-v2": ("gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv", "sparse"),
    "FetchSlideDense-v2": ("gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv", "dense"),
    "FetchPickAndPlace-v2": ("gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv", "sparse"),
    "FetchPickAndPlaceDense-v2": ("gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv", "dense"),
}


def ensure_fetch_v2_envs():
    """
    Newer gymnasium-robotics releases may expose Fetch v4 while rejecting the
    v2 ids. The experiments and trained operator checkpoints are keyed on v2,
    so register v2 ids locally instead of silently constructing v4 env ids.
    """
    for env_id, (entry_point, reward_type) in FETCH_V2_COMPAT_ENVS.items():
        if env_id in gym.registry:
            continue
        try:
            register(
                id=env_id,
                entry_point=entry_point,
                max_episode_steps=50,
                kwargs={"reward_type": reward_type},
            )
        except error.Error:
            if env_id not in gym.registry:
                raise


def make_fetch_env(env_id, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{env_id} is out of date.*", category=DeprecationWarning)
        return gym.make(env_id, **kwargs)


def norm_obs(obs):
    return np.concatenate([x.reshape(-1) for x in obs.values()]).astype(np.float32)


def operator_model_path(env_id, operator_model=None):
    if operator_model:
        return operator_model
    env_operator_model = os.environ.get("PMDC_OPERATOR_MODEL")
    if env_operator_model:
        return env_operator_model

    config_path = "./working_models.json"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Missing ./working_models.json. Provide the trained operator SAC checkpoint with "
            "--operator-model or PMDC_OPERATOR_MODEL."
        )

    with open(config_path, "r") as handle:
        models = json.load(handle)
    if env_id not in models:
        # Keep the local/remote setup on the v2 Fetch tasks; v1 is accepted
        # only as a legacy alias for older working_models.json entries.
        for version in ("v2", "v1"):
            fallback = env_id.rsplit("-", 1)[0] + f"-{version}"
            if fallback in models:
                return models[fallback]
        raise KeyError(f"No operator model configured for {env_id!r} in {config_path}")
    return models[env_id]


def _format_obs_reach(r_obs, o_obs):
    """FetchReach (10-dim, object-free) local-remote formatter.

    Emits the SAME 22-dim layout as the 25-dim Fetch tasks so the PMDC reward/recalibration
    indices ([0:3] = remote end-effector, [11:14] = operator end-effector) and the 22-dim
    observation space are unchanged -- no edits needed in PMDC_wrapper. Each arm contributes
    its full 10-dim FetchReach obs plus one pad dim, filling the 11 slots the object occupies
    in the standard layout.
    """
    r = r_obs['observation']  # [grip_pos(3), gripper_state(2), grip_velp(3), gripper_vel(2)]
    o = o_obs['observation']
    pad = np.zeros(1, dtype=r.dtype)
    r_obs['observation'] = np.concatenate([r[0:10], pad, o[0:10], pad]).astype(np.float32)
    r_obs['achieved_goal'] = r_obs['observation'][[0, 1, 2]].astype(np.float32)   # remote EE
    r_obs['desired_goal'] = o[[0, 1, 2]].astype(np.float32)                       # operator EE
    return r_obs


def format_obs(r_obs, o_obs):
    if r_obs['observation'].shape[0] == 10:  # FetchReach: object-free 10-dim layout
        return _format_obs_reach(r_obs, o_obs)

    def delete_idxs(a, idxs):
        for i in idxs:
            a = np.append(a[:i], a[i + 1:])
        return a

    r_obs['observation'] = delete_idxs(r_obs['observation'], [17, 18, 19])  # object velr
    r_obs['observation'] = delete_idxs(r_obs['observation'], [14, 15, 16])  # object velp
    r_obs['observation'] = delete_idxs(r_obs['observation'], [12, 13])  # object rot
    r_obs['observation'] = delete_idxs(r_obs['observation'], [6, 7, 8])  # object rel pos
    r_obs['observation'] = delete_idxs(r_obs['observation'], [3, 4, 5])  # object pos
    # Append operator state information
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[0, 1, 2]])  # Operator position
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[9, 10, 11]])  # Operator gripper state
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[20, 21]])  # Operator gripper velocity
    r_obs['observation'] = np.append(r_obs['observation'], o_obs['observation'][[22, 23, 24]])  # Operator velocity

    r_obs['achieved_goal'] = r_obs['observation'][[0, 1, 2]]  # Remote end-effector position
    r_obs['desired_goal'] = o_obs['observation'][[0, 1, 2]]  # Operator end-effector position

    for key in r_obs.keys():
        r_obs[key] = r_obs[key].astype(np.float32)

    return r_obs


class REnvPDNormObs(gym.Env):
    """
    Local-remote PD-controller environment for FetchPush.
    The RL agent outputs PD gains (P, D) and the error between remote and local
    end-effectors provides the reward signal.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, env_id="FetchPush-v2", seed=None, fname=None, operator_model=None):
        super(REnvPDNormObs, self).__init__()

        self.remote_Env = make_fetch_env(env_id)
        self.operator_Env = make_fetch_env(env_id)
        seed = np.random.randint(0, 100) if seed is None else seed
        self.fname = fname

        model_dir = operator_model_path(env_id, operator_model)
        self.operator = SAC.load(model_dir, env=self.remote_Env)

        self.prev_error = np.zeros(3)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.action_space = Box(low=-80, high=80, shape=(2,))
        self.rew_count = 0
        self.reward = 0

        # Initialize obs containers (populated in reset)
        self.r_obs = None
        self.o_obs = None

    def step(self, action):
        P, D = (action[0], action[1])
        error = self.r_obs['achieved_goal'] - self.r_obs['desired_goal']
        proportional = error
        derivative = error - self.prev_error

        pd_value = P * proportional + D * derivative
        self.r_obs, r_rew, r_term, r_trunc, r_info = self.remote_Env.step(np.append(pd_value, [0]))
        self.prev_error = error

        operator_action, _states = self.operator.predict(self.o_obs, deterministic=True)
        self.o_obs, o_rew, o_term, o_trunc, o_info = self.operator_Env.step(operator_action)

        self.r_obs = format_obs(self.r_obs, self.o_obs)

        reward = -np.linalg.norm(self.r_obs['achieved_goal'] - self.r_obs['desired_goal'])
        self.reward = reward
        self.rew_count += float(reward)
        return self.r_obs['observation'], float(reward), r_term, r_trunc, {"operator_reward": o_rew}

    def reset(self, **kwargs):
        self.rew_count = 0
        self.prev_error = np.zeros(3)
        self.r_obs, _ = self.remote_Env.reset(**kwargs)
        self.o_obs, _ = self.operator_Env.reset(**kwargs)
        return format_obs(self.r_obs, self.o_obs)['observation'], {}

    def render(self):
        return self.remote_Env.render()

    def close(self):
        self.operator_Env.close()
        self.remote_Env.close()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return float(-np.linalg.norm(achieved_goal - desired_goal))


class REnvDirectControl(REnvPDNormObs):
    """
    Local-remote FetchPush environment where the RL policy controls the remote
    Fetch action directly instead of outputting PD gains.

    The operator policy and reward are unchanged from REnvPDNormObs, so this
    isolates whether delay sensitivity comes from the PD-gain control
    abstraction or from delayed remote tracking itself.
    """

    def __init__(self, env_id="FetchPush-v2", seed=None, fname=None, operator_model=None):
        super().__init__(env_id=env_id, seed=seed, fname=fname, operator_model=operator_model)
        self.action_space = self.remote_Env.action_space

    def step(self, action):
        remote_action = np.asarray(action, dtype=np.float32)
        remote_action = np.clip(remote_action, self.action_space.low, self.action_space.high)
        self.r_obs, r_rew, r_term, r_trunc, r_info = self.remote_Env.step(remote_action)

        operator_action, _states = self.operator.predict(self.o_obs, deterministic=True)
        self.o_obs, o_rew, o_term, o_trunc, o_info = self.operator_Env.step(operator_action)

        self.r_obs = format_obs(self.r_obs, self.o_obs)
        self.prev_error = self.r_obs["achieved_goal"] - self.r_obs["desired_goal"]

        reward = -np.linalg.norm(self.prev_error)
        self.reward = reward
        self.rew_count += float(reward)
        return self.r_obs["observation"], float(reward), r_term, r_trunc, {"operator_reward": o_rew}


# Local-remote variants. The 25-dim/4-dim object tasks (FetchPush, FetchSlide,
# FetchPickAndPlace) share one format_obs() index layout; FetchReach (10-dim, no object) is
# handled by the _format_obs_reach() branch, which emits the same 22-dim formatted layout so
# the rest of the pipeline (PMDC reward indices, obs space) is unchanged.
LOCAL_REMOTE_ENVS = {
    "FetchPush-RemotePDNorm-v0": "FetchPush-v2",
    "FetchSlide-RemotePDNorm-v0": "FetchSlide-v2",
    "FetchPickAndPlace-RemotePDNorm-v0": "FetchPickAndPlace-v2",
    "FetchReach-RemotePDNorm-v0": "FetchReach-v2",
}

DIRECT_LOCAL_REMOTE_ENVS = {
    "FetchPush-RemoteDirect-v0": "FetchPush-v2",
}


def register_local_remote_envs():
    ensure_fetch_v2_envs()
    for remote_id, base_env_id in LOCAL_REMOTE_ENVS.items():
        try:
            gym.spec(remote_id)
        except Exception:
            register(
                id=remote_id,
                entry_point="robo_local_remote_env:REnvPDNormObs",
                max_episode_steps=50,
                kwargs={"env_id": base_env_id},
            )
    for remote_id, base_env_id in DIRECT_LOCAL_REMOTE_ENVS.items():
        try:
            gym.spec(remote_id)
        except Exception:
            register(
                id=remote_id,
                entry_point="robo_local_remote_env:REnvDirectControl",
                max_episode_steps=50,
                kwargs={"env_id": base_env_id},
            )


register_local_remote_envs()
