# Predictive-Model Delay-Correction Reinforcement Learning
![Alt Text](https://i.imgur.com/0a0fV5d.png)

**Abstract** Local-remote systems enable robots to perform complex tasks in hazardous environments such as space, and nuclear power stations. However, mapping positions between local and remote devices can be challenging due to time delays compromising system performance and stability. Improving the synchronicity and stability of local-remote systems is crucial to enabling robots to interact with environments at greater distances and under highly challenging network conditions (e.g. time delay). We propose an adaptive control method using reinforcement learning to address the problem of time delayed control. By adjusting controller parameters in real-time, this adaptive controller compensates for stochastic delays and improves synchronicity between local and remote robotic manipulators. 

To increase the performance of the adaptive PD controller, we develop a model-based reinforcement learning technique which efficiently incorporates multi-steps delays into the learning framework. Using the proposed technique the performance of the local-remote system is stabilised for stochastic communication time-delays up to 290ms. The results show that the proposed model-based reinforcement learning method outperforms Soft-Actor Critic and augmented state Soft-Actor Critic methods.

# Setup
`pip install -r ./requirements.txt`

# Run
`python delay_correcting_training.py`

<!-- # Applying PMDC to delayed environments
In order to apply PMDC you need to specify the ammount of delay to correct for e.g. 8 steps of action delay.
`
env = gym.make(env_id, seed=seed)
env = UnseenRandomDelayWrapper(env, obs_delay_range=range(0, 1), act_delay_range=range(ACT_DELAY-1, ACT_DELAY)
env = OLDUnDelayWrapper(), delay=OBS_DELAY+ACT_DELAY, env_id=env_id, pretrain=pretrain, n_models=n_models)


` -->

In order to use PMDC on custom environments you can simply wrap the environment in the PMDC wrapper which will train and correct for constant action delays. In order to handle stochastic delays call the `Augmented` wrapper with given stochastic range.

`
import AugmentedRandomDelayWrapper, UnseenRandomDelayWrapper from wrappers_rd
from PMDC_wrapper import PMDC

env = PMDC( UnseenRandomDelayWrapper (gym.make(env_id),
    obs_delay_range=range(0, 1),
    act_delay_range=range(ACT_DELAY-1, ACT_DELAY)), delay=OBS_DELAY+ACT_DELAY, env_id=env_id, n_models=n_models)
env = AugmentedRandomDelayWrapper(env, obs_delay_range=range(OBS_DELAY,OBS_DELAY+5), act_delay_range=range(0,1))
`

The implementation of the random delay wrapper used to implement random action and observation delays was modified from Bouteiller et al. "[Reinforcement Learning with Random Delays](https://openreview.net/forum?id=QFYnKlBJYR)" - [Arxiv](https://arxiv.org/abs/2010.02966) - [GitHub](https://github.com/rmst/rlrd)
