import numpy as np
from copy import deepcopy


def get_hamiltonian(conf, momenta, get_action, action_params):
	return np.sum(momenta ** 2) / 2.0 + get_action(conf, *action_params)


def run_trajectory(conf, get_action, get_force, get_random_momenta, action_params
	               num_steps, delta_tau):
	conf_old = deepcopy(conf)

	momenta = get_random_momenta(conf.shape)
	H_initial = get_hamiltonian(conf, momenta, get_action, action_params)

	for _ in num_steps:
		new_conf = conf + momenta * delta_tau
		new_momenta = momenta - delta_tau * get_force(conf, *action_params)
		conf = deepcopy(new_conf)
		momenta = deepcopy(new_momenta)

	H_final = get_hamiltonian(conf, momenta, get_action, action_params)
	if np.exp(H_initial - H_final) > np.random.random():
		return conf, True
	return conf_old, False