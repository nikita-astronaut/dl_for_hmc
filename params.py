import lattice_phi_four
import functools

force = lattice_phi_four.get_force
action = lattice_phi_four.get_action
get_random_conf = lattice_phi_four.get_random_conf
get_random_momenta = lattice_phi_four.get_random_momenta
shape = (25, 25)
action_params = [1.0, 0.1]
delta_tau = 0.01
num_steps = 100
low_acceptance = 0.65
high_acceptance = 0.75
tau_modification_factor = 1.1
initial_conf = None
n_trajectories = 2000
observables = [functools.partial(lattice_phi_four.get_n_moment_field, n = 1), functools.partial(lattice_phi_four.get_n_moment_field, n = 2), lattice_phi_four.get_std]