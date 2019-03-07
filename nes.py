import numpy as np
np.random.seed(1)

from nes_suppl import ACTIONS, ACTIONS_S, basis_functions, basis_functions_s


def act(obs, weights):
	global ACTIONS
	basis_funcs = basis_functions(obs)
	weights = weights.reshape((len(ACTIONS), -1))

	q_estimate = np.dot(weights, basis_funcs)
	action_i = np.argmax(q_estimate, axis=0)
	return  ACTIONS[action_i]


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
		observation, reward, done, info = env.step(*action)
		rollout_reward += reward

	return rollout_reward


def compute_gradients(m, cov_matrix, params):
	cov_matrix_inv = np.linalg.pinv(cov_matrix)
	prms_mean = (params - m)

	grad_m = np.dot(cov_matrix_inv, prms_mean)
	grad_cov = np.dot(0.5 * cov_matrix_inv, prms_mean, prms_mean.T, cov_matrix_inv) - 0.5 * cov_matrix_inv

	return grad_m, grad_cov


def run_nes(env):
	global ACTIONS
	episodes_n = 15000 
	w = np.random.randn(len(ACTIONS) * 40)  # number of basis functions
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
