from scipy import linalg as sl
import numpy as np
np.random.seed(1)

from nes_suppl import ACTIONS, ACTIONS_S, basis_functions, basis_functions_s


def act(obs, weights):
	global ACTIONS_S
	basis_funcs = basis_functions_s(obs)  # change depending on the env
	weights = weights.reshape((len(ACTIONS_S), -1))

	q_estimate = np.dot(weights, basis_funcs)
	action_i = np.argmax(q_estimate, axis=0)
	return  ACTIONS_S[action_i]


# fitness function
def f(env, params, render=False):
	observation = env.reset()
	rollout_reward = 0
	done = False
	while not done:
		if render:
			env.render()
		# depending on the current observation
		action = act(observation, params)
		observation, reward, done, info = env.step(action)
		rollout_reward += reward

	return rollout_reward


def compute_log_gradients(m, cov_matrix, params):
	cov_matrix_inv = np.linalg.pinv(cov_matrix)
	prms_mean = (params - m).T

	grad_m = np.dot(cov_matrix_inv, prms_mean)
	grad_cov = np.dot(np.dot(np.dot(0.5 * cov_matrix_inv, prms_mean), prms_mean.T), cov_matrix_inv) - 0.5 * cov_matrix_inv

	return grad_m.T, grad_cov


def compute_target_gradient(grads_m, grads_cov, fit_f_values, population_n):
	target_grad_m = 1 / population_n * np.sum(grads_m * fit_f_values.reshape((population_n, 1)), axis=0)
	target_grad_cov = 1 / population_n * np.sum(grads_cov * fit_f_values.reshape((population_n, 1, 1)), axis=0)
	return target_grad_m, target_grad_cov


def run_nes_improved(env):
	global ACTIONS_S
	episodes_n = 15000 
	w = np.random.randn(len(ACTIONS_S) * 10)  # number of basis functions
	population_n = 20
	cov_matrix = np.zeros((w.size, w.size))  # cov_matrix = a_matrix.T @ amatrix
	np.fill_diagonal(cov_matrix, 0.1)
	learning_rate = 1e-2

	for i in range(episodes_n):
		noise = np.random.randn(population_n, w.size)
		rewards = np.zeros(population_n)
		log_grads_m = np.zeros((population_n, w.size))
		log_grads_cov = np.zeros((population_n, w.size, w.size))

		for worker in range(population_n):
			worker_noise = (noise[worker]).T
			worker_params = w + (sl.sqrtm(cov_matrix) @ worker_noise).T
			rewards[worker] = f(env, worker_params, i > 5000)
			# print('Worker reward: {}'.format(rewards[worker]))

			log_grads_m[worker], log_grads_cov[worker] = compute_log_gradients(w, cov_matrix, worker_params)

		print('Episode: {} | Mean reward : {}'.format(i, np.mean(rewards)))

		target_grad_m, target_grad_cov = compute_target_gradient(log_grads_m, log_grads_cov, rewards, population_n)
		w = w + learning_rate * target_grad_m
		cov_matrix = cov_matrix + learning_rate * target_grad_cov


def run_nes(env):
	global ACTIONS_S
	episodes_n = 15000 
	w = np.random.randn(len(ACTIONS_S) * 10)  # number of basis functions
	population_n = 30
	sigma = 0.1
	learning_rate = 1e-2

	for i in range(episodes_n):
		noise = np.random.randn(population_n, w.size)
		rewards = np.zeros(population_n)

		for worker in range(population_n):
			worker_noise = noise[worker]
			worker_params = w + sigma * worker_noise
			rewards[worker] = f(env, worker_params, i > 5000)

		print('Episode: {} | Mean reward : {}'.format(i, np.mean(rewards)))

		normalized_rewards = (np.asarray(rewards) - np.mean(rewards)) / np.std(rewards)
		w = w + learning_rate / (population_n * sigma) * np.dot(noise.T, normalized_rewards)

	return -1
