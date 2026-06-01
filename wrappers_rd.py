from collections import deque
from random import sample
import itertools

import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Box

import numpy as np


class RandomDelayWrapper(gym.Wrapper):
    """
    Wrapper for any non-RTRL environment, modelling random observation and action delays
    NB: alpha refers to the observation delay, it is >= 0
    NB: The state-space now contains two different action delays:
        kappa is such that alpha+kappa is the index of the first action that was going to be applied when the observation started being captured, it is useful for the model
            (when kappa==0, it means that the delay is actually 1)
        beta is such that alpha+beta is the index of the last action that is known to have influenced the observation, it is useful for credit assignment (e.g. AC/DC)
            (alpha+beta is often 1 step bigger than the action buffer, and it is always >= 1)
    Kwargs:
        obs_delay_range: range in which alpha is sampled
        act_delay_range: range in which kappa is sampled
        initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
    """

    def __init__(
        self,
        env,
        obs_delay_range=range(0, 1),
        act_delay_range=range(0, 1),
        initial_action=None,
        skip_initial_actions=False,
        obs_buffer_len=None,
        act_buffer_len=None,
    ):
        super().__init__(env)
        self.wrapped_env = env
        self.obs_delay_range = obs_delay_range
        self.act_delay_range = act_delay_range
        self.obs_buffer_len = obs_delay_range.stop if obs_buffer_len is None else obs_buffer_len
        self.act_buffer_len = act_delay_range.stop if act_buffer_len is None else act_buffer_len
        if self.obs_buffer_len <= max(obs_delay_range):
            raise ValueError("obs_buffer_len must be larger than the largest observation delay")
        if self.act_buffer_len <= max(act_delay_range):
            raise ValueError("act_buffer_len must be larger than the largest action delay")

        self.observation_space = Tuple((
            env.observation_space,  # most recent observation
            Tuple([env.action_space] * self.action_history_len),  # action buffer
            Discrete(self.obs_buffer_len),  # observation delay int64
            Discrete(self.act_buffer_len),  # action delay int64
        ))

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_actions = deque(maxlen=self.obs_buffer_len + self.act_buffer_len)
        self.past_observations = deque(maxlen=self.obs_buffer_len)
        self.arrival_times_actions = deque(maxlen=self.act_buffer_len)
        self.arrival_times_observations = deque(maxlen=self.obs_buffer_len)

        self.t = 0
        self.done_signal_sent = False
        self.next_action = None
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0

    @property
    def action_history_len(self):
        return self.obs_buffer_len + self.act_buffer_len - 1

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0
        self.done_signal_sent = False
        first_observation, reset_info = self.env.reset(**kwargs)

        # fill up buffers
        self.t = - (self.obs_buffer_len + self.act_buffer_len)
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            self.send_action(act, init=True)
            self.send_observation((first_observation, 0., False, {}, 0, 1))
            self.t += 1
        self.receive_action()  # an action has to be applied

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        return received_observation, {}

    def step(self, action):
        """
        When kappa is 0 and alpha is 0, this is equivalent to the RTRL setting
        """
        # at the brain
        self.send_action(action)

        # at the remote actor
        if self.t < self.act_delay_range.stop and self.skip_initial_actions:
            self.receive_action()
        elif self.done_signal_sent:
            self.send_observation(self.past_observations[0])
        else:
            m, r, terminated, truncated, info = self.env.step(self.next_action)
            d = terminated or truncated
            kappa, beta = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = d
            self.send_observation((m, self.cum_rew_actor, d, info, kappa, beta))

        # at the brain again
        m, cum_rew_actor_delayed, d, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        self.t += 1
        return m, r, d, False, info

    def send_action(self, action, init=False):
        kappa, = sample(self.act_delay_range, 1) if not init else [0, ]
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)

    def receive_action(self):
        prev_action_idx = self.prev_action_idx + 1
        next_action_idx = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.prev_action_idx = next_action_idx
        self.next_action = self.past_actions[next_action_idx]
        return next_action_idx, prev_action_idx

    def send_observation(self, obs):
        alpha, = sample(self.obs_delay_range, 1)
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        m, r, d, info, kappa, beta = self.past_observations[alpha]
        return (m, tuple(itertools.islice(self.past_actions, 0, self.past_actions.maxlen - 1)), alpha, kappa, beta), r, d, info


class UnseenRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that translates the RandomDelayWrapper back to the usual RL setting.
    Use this wrapper to see what happens to vanilla RL algorithms facing random delays.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        t, info = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return t[0], info

    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return (t[0], *aux)


class AugmentedRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that augments the observation with the action history to account for delays.
    """

    def __init__(self, env, delay_v=None, **kwargs):
        super().__init__(env, **kwargs)
        self.flat_observation_dim = env.observation_space.shape[0]
        self.action_buffer_dim = env.action_space.shape[0] * self.action_history_len
        if delay_v is not None:
            # Backward compatible with the original caller's 8 + delay_v shape convention.
            expected_dim = 8 + delay_v
            computed_dim = self.flat_observation_dim + self.action_buffer_dim
            if expected_dim != computed_dim:
                raise ValueError(
                    f"delay_v gives observation dim {expected_dim}, but the configured delay buffers "
                    f"require {computed_dim}"
                )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.flat_observation_dim + self.action_buffer_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        t, info = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        aug_state = np.append(t[0], t[1])
        return aug_state, info

    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        aug_state = np.append(t[0], t[1])
        return (aug_state, *aux)


class AugmentedDelayInfoWrapper(AugmentedRandomDelayWrapper):
    """
    Augments observations with action history and the sampled delay values.

    The appended delay values are alpha, kappa, and beta from RandomDelayWrapper:
    observation delay, current action delay index, and previous action influence index.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        dim = self.observation_space.shape[0] + 3
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def _augment(self, t):
        obs, actions, alpha, kappa, beta = t
        delay_values = np.asarray([alpha, kappa, beta], dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(obs, dtype=np.float32).reshape(-1),
                np.asarray(actions, dtype=np.float32).reshape(-1),
                delay_values,
            ]
        )

    def reset(self, **kwargs):
        t, info = RandomDelayWrapper.reset(self, **kwargs)
        return self._augment(t), info

    def step(self, action):
        t, *aux = RandomDelayWrapper.step(self, action)
        return (self._augment(t), *aux)


def simple_wifi_sampler1():
    return np.random.choice([1, 2, 3, 4, 5, 6], p=[0.3082, 0.5927, 0.0829, 0.0075, 0.0031, 0.0056])


def simple_wifi_sampler2():
    return np.random.choice([1, 2, 3, 4], p=[0.3082, 0.5927, 0.0829, 0.0162])


class WifiDelayWrapper1(RandomDelayWrapper):
    def __init__(self, env, initial_action=None, skip_initial_actions=False):
        super().__init__(env, obs_delay_range=range(0, 7), act_delay_range=range(0, 7), initial_action=initial_action, skip_initial_actions=skip_initial_actions)

    def send_observation(self, obs):
        alpha = simple_wifi_sampler1()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def send_action(self, action, init=False):
        kappa = simple_wifi_sampler1() if not init else 0
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)


class WifiDelayWrapper2(RandomDelayWrapper):
    def __init__(self, env, initial_action=None, skip_initial_actions=False):
        super().__init__(env, obs_delay_range=range(0, 5), act_delay_range=range(0, 5), initial_action=initial_action, skip_initial_actions=skip_initial_actions)

    def send_observation(self, obs):
        alpha = simple_wifi_sampler2()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def send_action(self, action, init=False):
        kappa = simple_wifi_sampler2() if not init else 0
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)
