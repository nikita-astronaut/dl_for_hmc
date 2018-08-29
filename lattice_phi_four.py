import numpy as np

def get_action(configuration, kappa, lamb):
	'''
	configuration: D-dimentional python array with the configuration
	kappa: action parameter
	lamb: action parameter lambda
	'''
	ndims = len(configuration.shape)
	action = 0

	# kinetic term sums over all dimention derivatives
	for dim in range(ndims):
		action += -2.0 * kappa * np.sum(configuration * np.roll(configuration, shift = 1, axis = dim))
	action += np.sum(configuration ** 2)
	action += lamb * np.sum((configuration ** 2 - 1.0) ** 2 - 1.0)
	return action
