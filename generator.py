import params
import metropolis
import matplotlib.pyplot as plt
import numpy as np

force = params.force
action = params.action
shape = params.shape
action_params = params.action_params
delta_tau = params.delta_tau
num_steps = params.num_steps
low_acceptance = params.low_acceptance
high_acceptance = params.high_acceptance
tau_modification_factor = params.tau_modification_factor
initial_conf = params.initial_conf
get_random_conf = params.get_random_conf
get_random_momenta = params.get_random_momenta
n_trajectories = params.n_trajectories
observables = params.observables


def print_observables(conf, observables):
	res = ''
	for obs in observables:
		value, name = obs(conf)
		res += name + ' = ' + str(value) + ', '
	print(res)


def correlation_times(history):
	sigma = np.std(history)
	mean = history.mean()
	corr = []
	for n in range(100):
		arr1 = history[:len(history)-n]
		arr2 = history[n:]
		corr.append(np.mean(arr1 * arr2) - arr1.mean() * arr2.mean())
	return np.abs(np.array(corr) / corr[0])

if initial_conf == None:
	conf = get_random_conf(shape, mode='hot')
else:
	conf = np.load(initial_conf)

acceptance_history = []
action_history = []

for n_tr in range(n_trajectories):
	conf, accepted, dH = metropolis.run_trajectory(conf, action, force, get_random_momenta, action_params, num_steps, delta_tau)
	acceptance_history.append(accepted)
	print(accepted, dH, action(conf, *action_params) / shape[0] /shape[1], -(-1 + 4.0 * action_params[0] + 2.0 * action_params[1]) ** 2 / 4.0 / action_params[1])
	print_observables(conf, observables)
	action_history.append(conf.mean())
	print(np.sqrt(1.0 + (-1.0 + 4.0 * action_params[0]) / 2.0 / action_params[1]))

taus = correlation_times(np.array(action_history)[1000:])
plt.scatter(np.arange(0, taus.shape[0]), taus)
plt.xlabel('$\\tau,$ trajectories', fontsize = 14)
plt.ylabel('$C(\\tau) / C(0)$ normalized autocorrelation', fontsize = 14)
plt.grid(True, alpha = 0.5, linestyle='--')
plt.savefig('./plots/autocorrelation.pdf')
plt.show()