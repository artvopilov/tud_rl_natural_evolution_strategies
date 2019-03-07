from scipy import linalg as sl
import threading
import gym
import numpy as np
np.random.seed(1)

from nes_suppl import ACTIONS, ACTIONS_S, basis_functions, basis_functions_s


def act(obs, weights, actions):
	basis_funcs = basis_functions_s(obs)  # change depending on the env
	weights = weights.reshape((len(actions), -1))

	q_estimate = np.dot(weights, basis_funcs)
	action_i = np.argmax(q_estimate, axis=0)
	return  actions[action_i]


# fitness function
def f(env_name, params, actions, rewards, i, render=False):
	env = gym.make(env_name)
	observation = env.reset()
	rollout_reward = 0
	done = False
	while not done:
		if render:
			env.render()
		# depending on the current observation
		action = act(observation, params, actions)
		observation, reward, done, info = env.step(action)
		rollout_reward += reward

	# print('Worker reward: {}'.format(rollout_reward))
	rewards[i] = rollout_reward
	# return rollout_reward


def compute_log_gradients(m, cov_matrix, params):
	cov_matrix_inv = np.linalg.inv(cov_matrix)
	prms_mean = (params - m).reshape((m.size, 1))
	# print(prms_mean.shape)

	grad_m = np.dot(cov_matrix_inv, prms_mean)
	grad_cov = 0.5 * np.dot(np.dot(np.dot(cov_matrix_inv, prms_mean), prms_mean.T), cov_matrix_inv) - 0.5 * cov_matrix_inv
	# print(np.dot(np.dot(cov_matrix_inv, prms_mean), prms_mean.T))
	# print(np.dot(cov_matrix_inv, prms_mean).shape)
	return grad_m.T, grad_cov


def compute_target_gradient(grads_m, grads_cov, fit_f_values, population_n):
	# print(grads_cov)
	# print(grads_m)
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
			# rewards[worker] = f(env, worker_params, ACTIONS_S, i > 5000)
			log_grads_m[worker], log_grads_cov[worker] = compute_log_gradients(w, cov_matrix, worker_params)

			t = threading.Thread(target=f, args=(env_name, worker_params, ACTIONS_S, rewards, worker, False))
			threads.append(t)
			t.start()

		for t in threads:
			t.join()

		print('Episode: {} | Mean reward : {}'.format(i, np.mean(rewards)))

		target_grad_m, target_grad_cov = compute_target_gradient(log_grads_m, log_grads_cov, rewards, population_n)
		# print('Gradients: ')
		# print(target_grad_m)  # 0 ????
		# print(target_grad_cov)
		w = w + learning_rate * target_grad_m
		cov_matrix = cov_matrix + learning_rate * target_grad_cov
		# print('Mean: ')
		# print(w)
		# print('Covariance matrix: ')
		# print(cov_matrix)


def run_nes(env_name):
	global ACTIONS_S
	episodes_n = 15000 
	w = np.random.randn(len(ACTIONS_S) * 9)  # number of basis functions
	population_n = 30
	sigma = 0.1
	learning_rate = 1e-2

	for i in range(episodes_n):
		noise = np.random.randn(population_n, w.size)
		rewards = np.zeros(population_n)

		threads = []
		for worker in range(population_n):
			worker_noise = noise[worker]
			worker_params = w + sigma * worker_noise
			# rewards[worker] = f(env_name, worker_params, ACTIONS_S, i > 10000)
			t = threading.Thread(target=f, args=(env_name, worker_params, ACTIONS_S, rewards, worker, False))
			threads.append(t)
			t.start()
		for t in threads:
			t.join()

		print('Rewards: {}'.format(rewards))
		print('Episode: {} | Mean reward : {}'.format(i, np.mean(rewards)))

		normalized_rewards = (np.asarray(rewards) - np.mean(rewards)) / np.std(rewards)
		w = w + learning_rate / (population_n * sigma) * np.dot(noise.T, normalized_rewards)

	return -1
