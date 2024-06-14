from jax import random
import jax.numpy as jnp

# initialize weights and biases of a single network
def init_single_net(parent_key, layer_widths):
    params = []
    keys = random.split(parent_key, num=len(layer_widths) - 1)
    # create the weights and biases for the network
    for in_dim, out_dim, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = random.split(key)
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * xavier_stddev]
        )
    return params


# generate weights and biases for all networks required in the problem
def init_MLP(parent_key, n_hl, n_unit):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''
    # set the neural network shape for u, v, h
    layers1 = [2] + n_hl * [n_unit] + [3]
    # set the neural network shape for mu
    layers2 = [2] + n_hl * [n_unit] + [1]

    # generate the random key for each network
    keys = random.split(parent_key, 2)
    # generate weights and biases for
    params_u = init_single_net(keys[0], layers1)
    params_mu = init_single_net(keys[0], layers2)
    return [params_u, params_mu]
