import quanser_robots
import gym
from gym.wrappers.monitor import Monitor

import numpy as np
from nes import run_nes, run_nes_improved


env_name = 'Pendulum-v0'  # BallBalancerSim-v0 
# print(env.action_space.low, env.action_space.high)
# print(env.observation_space.low, env.observation_space.high)
# print(env.action_space.sample())

run_nes_improved(env_name)

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
