import numpy as np
from copy import deepcopy


def get_hamiltonian(conf, momenta, get_action, action_params):
	return np.sum(momenta ** 2) / 2.0 + get_action(conf, *action_params)


def run_trajectory(conf, get_action, get_force, get_random_momenta, action_params,
	               num_steps, delta_tau):
	conf_old = deepcopy(conf)

	momenta = get_random_momenta(conf.shape)
	H_initial = get_hamiltonian(conf, momenta, get_action, action_params)

	# leapfrog integrator
	for _ in range(num_steps):
		momenta = momenta - delta_tau * get_force(conf, *action_params) / 2.0
		conf = conf + momenta * delta_tau
		momenta = momenta - delta_tau * get_force(conf, *action_params) / 2.0

	H_final = get_hamiltonian(conf, momenta, get_action, action_params)

	if np.exp(H_initial - H_final) > np.random.random():
		return conf, True, H_final - H_initial
	return conf_old, False, H_final - H_initial

