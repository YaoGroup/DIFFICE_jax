from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_map

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


# generate weights and biases for all networks required in the XPINNs problem
def init_nets(parent_key, n_hl, n_unit, n_sub=1, aniso=False):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    :param n_sub: number of sub-regions
    '''
    # set the default number of output for viscosity
    n_mu = 1
    # for anisotropic model
    if aniso:
        # number of viscosity output is 2
        n_mu = 2

    # set the neural network shape for u, v, h
    layers1 = [2] + n_hl * [n_unit] + [3]
    # set the neural network shape for mu
    layers2 = [2] + n_hl * [n_unit] + [n_mu]

    # generate the random key for each network (in list format)
    _, *keys = random.split(parent_key, 2*n_sub+1)

    # generate weights and biases for each group
    params_u = tree_map(lambda x: init_single_net(x, layers1), keys[0:n_sub])
    params_mu = tree_map(lambda x: init_single_net(x, layers2), keys[n_sub:])

    return dict(net_u=params_u, net_mu=params_mu)
