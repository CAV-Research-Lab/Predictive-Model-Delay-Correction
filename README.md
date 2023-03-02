# Predictive-Model Delay-Correction Reinforcement Learning
![Alt Text](https://i.imgur.com/0a0fV5d.png)

**Abstract** Local-remote systems enable robots to perform complex tasks in hazardous environments such as space, and nuclear power stations. However, mapping positions between local and remote devices can be challenging due to time delays compromising system performance and stability. Improving the synchronicity and stability of local-remote systems is crucial to enabling robots to interact with environments at greater distances and under highly challenging network conditions (e.g. time delay). We propose an adaptive control method using reinforcement learning to address the problem of time delayed control. By adjusting controller parameters in real-time, this adaptive controller compensates for stochastic delays and improves synchronicity between local and remote robotic manipulators. 

To increase the performance of the adaptive PD controller, we develop a model-based reinforcement learning technique which efficiently incorporates multi-steps delays into the learning framework. Using the proposed technique the performance of the local-remote system is stabilised for stochastic communication time-delays up to 290ms. The results show that the proposed model-based reinforcement learning method outperforms Soft-Actor Critic and augmented state Soft-Actor Critic methods.

# Setup
