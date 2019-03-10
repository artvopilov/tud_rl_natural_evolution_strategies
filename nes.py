from scipy import linalg as sl
import threading
import gym
import torch
import numpy as np
import pickle

from nes_suppl import ACTIONS, basis_functions
from nes_model import ModelNes
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def act_basis_funcs(obs, weights, actions):
	basis_funcs = basis_functions(obs)  # change depending on the env
	weights = weights.reshape((len(actions), -1))

	q_estimate = np.dot(weights, basis_funcs)
	action_i = np.argmax(q_estimate, axis=0)
	return actions[action_i]


def act_nn(obs, weights, actions):
	model = ModelNes(obs.size, len(actions))
	vector_to_parameters(torch.from_numpy(weights).float(), model.parameters())
	with torch.no_grad():
		q_estimate = model(torch.from_numpy(obs).float())
	action_i = np.argmax(q_estimate.data.numpy())
	return actions[action_i]


# fitness function
def f(env, params, actions, render=False):
	observation = env.reset()
	rollout_reward = 0
	done = False
	while not done:
		if render:
			env.render()
		# with basis functions
		action = act_basis_funcs(observation, params, actions)

		# with nn
		# action = act_nn(observation, params, actions)

		observation, reward, done, info = env.step(action[0])
		rollout_reward += reward

	# rewards[i] = rollout_reward
	return rollout_reward


def compute_log_gradients(m, cov_matrix, params):
	cov_matrix_inv = np.linalg.inv(cov_matrix)
	prms_mean = (params - m).reshape((m.size, 1))

	grad_m = np.dot(cov_matrix_inv, prms_mean)
	grad_cov = 0.5 * np.dot(np.dot(np.dot(cov_matrix_inv, prms_mean), prms_mean.T), cov_matrix_inv) - 0.5 * cov_matrix_inv
	return grad_m.T, grad_cov


def compute_target_gradient(grads_m, grads_cov, fit_f_values, population_n):
	target_grad_m = 1 / population_n * np.sum(grads_m * fit_f_values.reshape((population_n, 1)), axis=0)
	target_grad_cov = 1 / population_n * np.sum(grads_cov * fit_f_values.reshape((population_n, 1, 1)), axis=0)
	return target_grad_m, target_grad_cov


def run_nes_improved(env_name):
	global ACTIONS_S
	episodes_n = 15000 
	w = np.random.randn(len(ACTIONS_S) * 9)  # number of basis functions
	population_n = 24
	cov_matrix = np.zeros((w.size, w.size))  # cov_matrix = a_matrix.T @ amatrix
	np.fill_diagonal(cov_matrix, 0.01)
	learning_rate = 1e-2

	for i in range(episodes_n):
		noise = np.random.randn(population_n, w.size)
		rewards = np.zeros(population_n)
		log_grads_m = np.zeros((population_n, w.size))
		log_grads_cov = np.zeros((population_n, w.size, w.size))

		threads = []
		for worker in range(population_n):
			worker_noise = (noise[worker]).T
			worker_params = w + (sl.sqrtm(cov_matrix) @ worker_noise).T  # np.sqrt(cov_matrix)
			log_grads_m[worker], log_grads_cov[worker] = compute_log_gradients(w, cov_matrix, worker_params)

			t = threading.Thread(target=f, args=(env_name, worker_params, ACTIONS_S, rewards, worker, False))
			threads.append(t)
			t.start()

		for t in threads:
			t.join()

		print('Episode: {} | Mean reward : {}'.format(i, np.mean(rewards)))

		target_grad_m, target_grad_cov = compute_target_gradient(log_grads_m, log_grads_cov, rewards, population_n)
		w = w + learning_rate * target_grad_m
		cov_matrix = cov_matrix + learning_rate * target_grad_cov


def run_nes(env, episodes_n=10000, save_checkpoints=True):
	global ACTIONS

	# env = gym.make(env_name)
	if save_checkpoints:
		with open('best_weights_sync.json', 'rb') as bw_f:
			saved_w = pickle.load(bw_f)
		w = saved_w
	else:
		w = np.random.randn(len(ACTIONS) * 38)  # number of basis functions

	# for implementation with nn
	# model = ModelNes(8, len(ACTIONS))  # dimension of observation
	# # Remove grad from model
	# for param in model.parameters():
	# 	param.requires_grad = False
	# w = saved_w  # parameters_to_vector(model.parameters()).numpy()
	
	population_n = 20
	sigma = 0.1
	learning_rate = 1e-3

	highest_reward = 550
	mean_reward = 0

	for i in range(episodes_n):
		noise = np.random.randn(population_n, len(w))
		rewards = np.zeros(population_n)

		# threads = []
		for worker in range(population_n):
			worker_noise = noise[worker]
			worker_params = w + sigma * worker_noise
			rewards[worker] = f(env, worker_params, ACTIONS)
		# 	t = threading.Thread(target=f, args=(env, worker_params, ACTIONS, rewards, worker, False))
		# 	threads.append(t)
		# 	t.start()
		# for t in threads:
		# 	t.join()

		mean_reward = np.mean(rewards)
		print('Episode: {} | Mean reward : {}'.format(i, mean_reward))
		normalized_rewards = (rewards - mean_reward) / np.std(rewards)
		w = w + learning_rate / (population_n * sigma) * np.dot(noise.T, normalized_rewards).T

		if save_checkpoints and mean_reward > highest_reward:
			highest_reward = mean_reward
			with open('best_weights_sync.json', 'wb') as file_bw:
				print(w)
				pickle.dump(w, file_bw)


	return lambda obs: act_basis_funcs(obs, w, ACTIONS)[0]
