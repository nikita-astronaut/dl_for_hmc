import numpy as np

def get_action(configuration, kappa, lamb):
	'''
	calculates action, see Montvay&Munster formula (2.9)
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

def get_force(configuration, kappa, lamb):
	'''
	calculates force of the action $f_x = \partial S / \partial \phi_x$
	configuration: D-dimentional python array with the configuration
	kappa: action parameter
	lamb: action parameter lambda
	'''
	force = 2 * configuration
	ndims = len(configuration.shape)

	for dim in range(ndims):
		force += -2 * kappa * (np.roll(configuration, shift = 1, axis = dim) + np.roll(configuration, shift = -1, axis = dim))
	force += 4 * lamb * (configuration ** 3 - configuration)
	return force
