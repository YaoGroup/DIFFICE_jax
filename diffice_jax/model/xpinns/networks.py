import sys
import jax.numpy as jnp
from jax import lax
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from equation.ssa_eqn_iso import vectgrad

# define the basic formation of neural network
def neural_net(params, x, scl, act_s=0):
    '''
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param sgn:  1 for even function and -1 for odd function
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    # choose the activation function
    actv = [jnp.tanh, jnp.sin][act_s]
    # normalize the input
    H = x  # input has been normalized
    # separate the first, hidden and last layers
    first, *hidden, last = params
    # calculate the first layers output with right scale
    H = actv(jnp.dot(H, first[0]) * scl + first[1])
    # calculate the middle layers output
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    # no activation function for last layer
    var = jnp.dot(H, last[0]) + last[1]
    return var


# wrapper to create solution function with given domain size
def solu_create(scale, scl=1, act_s=0):
    '''
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    '''
    def f(params, x, idx):
        # generate the NN
        uvh = neural_net(params['net_u'][idx], x, scl, act_s)
        mu = neural_net(params['net_mu'][idx], x, scl, act_s)
        sol = jnp.hstack([uvh, jnp.exp(mu)])
        return sol

    def gradf(params, x, idx):
        drange = scale[idx][1]
        lx0, ly0, u0, v0 = drange[0:4]
        u0m = lax.max(u0, v0)
        l0m = lax.max(lx0, ly0)
        ru0 = u0 / u0m
        rv0 = v0 / u0m
        rx0 = lx0 / l0m
        ry0 = ly0 / l0m
        coeff = jnp.hstack([ru0/rx0, ru0/ry0, rv0/rx0, rv0/ry0, 1/rx0, 1/ry0])
        # load the network
        net = lambda x: f(params, x, idx)
        # calculate the gradient
        grad = vectgrad(net, x)[0]
        # ensure that the velocity gradient is normalize by the same scale
        # (this is an important step to compute the normalized strain rate)
        duvh = grad[:, 0:6] * coeff
        # calculate the strain rate
        u_x = duvh[:, 0:1]
        u_y = duvh[:, 1:2]
        v_x = duvh[:, 2:3]
        v_y = duvh[:, 3:4]
        strate = (u_x ** 2 + v_y ** 2 + 0.25 * (u_y + v_x) ** 2 + u_x * v_y) ** 0.5
        # group the solution
        gsol = jnp.hstack([duvh, strate])
        return gsol

    return f, gradf