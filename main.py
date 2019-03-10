import quanser_robots
import gym
from gym.wrappers.monitor import Monitor

import numpy as np
from nes import run_nes, run_nes_improved
from torch.nn.utils import parameters_to_vector

from nes_new import run_nes_new


env_name = 'BallBalancerSim-v0'  # Pendulum-v0
# env = gym.make(env_name)
# print(env.action_space.low, env.action_space.high)
# print(env.observation_space.low, env.observation_space.high)
# print(env.observation_space.sample().size)

# from nes_model import ModelNes
# model = ModelNes(3, 5)
# for p in model.parameters():
# 	p.requires_grad = False

# print(sum(p.numel() for p in model.parameters()))

# print(parameters_to_vector(model.parameters()).numpy().size)

run_nes(gym.make(env_name))
# run_nes_improved(env_name)

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
