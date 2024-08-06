"""
@author: Yongji Wang
"""

import jax.numpy as jnp
from jax import vjp, vmap, lax


# generate matrix required for vjp for vector gradient
def vgmat(x, n_out, idx=None):
    '''
    :param n_out: number of output variables
    :param idx: indice (list) of the output variable to take the gradient
    '''
    if idx is None:
        idx = range(n_out)
    # obtain the number of index
    n_idx = len(idx)
    # obtain the number of input points
    n_pt = x.shape[0]
    # determine the shape of the gradient matrix
    mat_shape = [n_idx, n_pt, n_out]
    # create the zero matrix based on the shape
    mat = jnp.zeros(mat_shape)
    # choose the associated element in the matrix to 1
    for l, ii in zip(range(n_idx), idx):
        mat = mat.at[l, :, ii].set(1.)
    return mat


# vector gradient of the output with input
def vectgrad(func, x):
    # obtain the output and the gradient function
    sol, vjp_fn = vjp(func, x)
    # determine the mat grad
    mat = vgmat(x, sol.shape[1])
    # calculate the gradient of each output with respect to each input
    grad0 = vmap(vjp_fn, in_axes=0)(mat)[0]
    # calculate the total partial derivative of output with input
    n_pd = x.shape[1] * sol.shape[1]
    # reshape the derivative of output with input
    grad = grad0.transpose(1, 0, 2)
    grad_all = grad.reshape(x.shape[0], n_pd)
    return grad_all, sol


def gov_eqn(net, x, scale):
    """
    :param net: the neural net instance for calculating the informed part
    """
    # setting the global parameters
    rho = 917
    rho_w = 1030
    gd = 9.8*(1-rho/rho_w)  # gravitational acceleration

    dmean, drange = scale[0:2]
    lx0, ly0, u0, v0 = drange[0:4]
    h0 = dmean[4]

    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    ru0 = u0 / u0m
    rv0 = v0 / u0m
    rx0 = lx0 / l0m
    ry0 = ly0 / l0m

    def grad1stOrder(net, x):
        grad, sol = vectgrad(net, x)
        h = sol[:, 2:3]
        mu = sol[:, 3:4]
        eta = sol[:, 4:5]

        u_x = grad[:, 0:1] * ru0 / rx0
        u_y = grad[:, 1:2] * ru0 / ry0
        v_x = grad[:, 2:3] * rv0 / rx0
        v_y = grad[:, 3:4] * rv0 / ry0
        h_x = grad[:, 4:5] / rx0
        h_y = grad[:, 5:6] / ry0
        strate = (u_x ** 2 + v_y ** 2 + 0.25 * (u_y + v_x) ** 2 + u_x * v_y) ** 0.5

        term1_1 = 2 * h * (mu * u_x + eta * (u_x + v_y))
        term2_1 = 2 * h * (mu * v_y + eta * (u_x + v_y))
        term12_2 = mu * h * (u_y + v_x)
        term1_3 = h * h_x
        term2_3 = h * h_y
        return jnp.hstack([term1_1, term2_1, term12_2, term1_3, term2_3, strate])

    func_g = lambda x: grad1stOrder(net, x)
    grad_term, term = vectgrad(func_g, x)

    e1term1 = grad_term[:, 0:1] / rx0  # (term1_1, x)
    e1term2 = grad_term[:, 5:6] / ry0  # (term12_2, y)
    e2term1 = grad_term[:, 3:4] / ry0  # (term2_1, y)
    e2term2 = grad_term[:, 4:5] / rx0  # (term12_2, x)
    e1term3 = term[:, 3:4]
    e2term3 = term[:, 4:5]
    strate = term[:, 5:6]

    e1 = e1term1 + e1term2 - e1term3
    e2 = e2term1 + e2term2 - e2term3

    f_eqn = jnp.hstack([e1, e2])
    val_term = jnp.hstack([e1term1, e1term2, e1term3, e2term1, e2term2, e2term3, strate])
    return f_eqn, val_term


#%%

def front_eqn(net, x, nb, scale):
    """
    :param net: the neural net instance for calculating the informed part
    :param nb: outward normal direction at the boundary
    """

    # setting the global parameters
    rho = 917
    rho_w = 1030
    gd = 9.8 * (1 - rho / rho_w)  # gravitational acceleration

    dmean, drange = scale[0:2]
    lx0, ly0, u0, v0 = drange[0:4]
    h0 = dmean[4]

    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    ru0 = u0 / u0m
    rv0 = v0 / u0m
    rx0 = lx0 / l0m
    ry0 = ly0 / l0m

    grad, sol = vectgrad(net, x)
    h = sol[:, 2:3]
    mu = sol[:, 3:4]
    eta = sol[:, 4:5]

    u_x = grad[:, 0:1] * ru0 / rx0
    u_y = grad[:, 1:2] * ru0 / ry0
    v_x = grad[:, 2:3] * rv0 / rx0
    v_y = grad[:, 3:4] * rv0 / ry0

    term1_1 = 2 * (mu * u_x + eta * (u_x + v_y))
    term2_1 = 2 * (mu * v_y + eta * (u_x + v_y))
    term12 = mu * (u_y + v_x)
    term_h = 0.5 * h

    e1 = term1_1 * nb[:, 0:1] + term12 * nb[:, 1:2] - term_h * nb[:, 0:1]
    e2 = term12 * nb[:, 0:1] + term2_1 * nb[:, 1:2] - term_h * nb[:, 1:2]

    f_eqn = jnp.hstack([e1, e2])
    val_term = jnp.hstack([term1_1, term2_1, term12, term_h])
    return f_eqn, val_term
