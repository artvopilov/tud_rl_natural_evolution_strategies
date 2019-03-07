import quanser_robots
import gym
from gym.wrappers.monitor import Monitor

import numpy as np
from nes import run_nes


env = gym.make('BallBalancerSim-v0')  # Pendulum-v0
# print(env.action_space.low, env.action_space.high)
# print(env.observation_space.low, env.observation_space.high)
# print(env.action_space.sample())

run_nes(env)

"""
for i_episode in range(3):
	observation = env.reset()
	for t in range(10000):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break"""