"""
@author: Yongji Wang
"""

import sys
import jax.numpy as jnp
import optax
from jax import random, jit, grad
import jax.flatten_util as flat_utl
from jax.debug.callback import call
from tensorflow_probability.substrates import jax as tfp
import functools

# create the Adam minimizer
@functools.partial(jit, static_argnames=("lossf", "opt"))
def adam_minimizer(lossf, params, data, opt, opt_state):
    """Basic gradient update step based on the opt optimizer."""
    grads, loss_info = grad(lossf, has_aux=True)(params, data)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss_info, opt_state


def adam_optimizer(key, lossf, params, dataf, epoch, lr=1e-3, aniso=False, schdul=None):
    """using the adam optimizer for the training.

    Args:
      key: random key
      lossf: loss function
      params: parameters of the networks
      dataf: data sampling function
      epoch: total number of iterations
      lr: learning rate
      aniso: whether the training is for anisotropic viscosity [boolean]
      schdul: scheduler function for modifying the weight of regularization term [callable]

    Returns:
      params: trained parameters of the networks
      loss_all: history of loss info over the training
    """
    if schdul is None:
        schdul = lambda x: 1.0
    # extract the initial wsp value if exist
    if hasattr(lossf, 'wsp'):
        wsp0 = lossf.wsp
    else:
        wsp0 = jnp.nan
    # select the Adam as the minimizer
    opt_Adam = optax.adam(learning_rate=lr)
    # obtain the initial state of the params
    opt_state = opt_Adam.init(params)
    # pre-allocate the loss variable
    loss_all = []
    nc = jnp.int32(jnp.round(epoch / 5))
    # start the training iteration
    for step in range(epoch):
        # split the new key for randomization
        key = random.split(key, 1)[0]
        # re-sampling the data points
        data = dataf(key)
        # minimize the loss function using Adam
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        # print the loss for every 100 iteration
        if (step+1) % 100 == 0:
            # print the results
            print(f"Step: {step+1} | Loss: {loss_info[0]:.4e} | Loss_d: {loss_info[1]:.4e} |"
            f" Loss_e: {loss_info[2]:.4e} | Loss_b: {loss_info[3]:.4e}", file=sys.stderr)
            # if for anisotropic training
            if aniso:
                # modify the wsp value over the iteration
                lossf.wsp = wsp0 * schdul(step+1)

        # saving the loss
        loss_all.append(loss_info[0:4])

    # obtain the total loss in the last iterations
    lossend = jnp.array(loss_all[-nc:])[:, 0]
    # find the minimum loss value
    lmin = jnp.min(lossend)
    # optain the last loss value
    llast = lossend[-1]
    # guarantee the loss value in last iteration is smaller than anyone before
    while llast > lmin:
        # split the new key for randomization
        key = random.split(key, 1)[0]
        # re-sampling the data points
        data = dataf(key)
        # minimize the loss function using Adam
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        llast = loss_info[0]
        # saving the loss
        loss_all.append(loss_info[0:4])

    return params, loss_all


# A factory to create a function required by tfp.optimizer.lbfgs_minimize.
def lbfgs_function(lossf, init_params, data):
    # obtain the 1D parameters and the function that can turn back to the pytree
    _, unflat = flat_utl.ravel_pytree(init_params)

    def update(params_1d):
        # updating the model's parameters from the 1D array
        params = unflat(params_1d)
        return params

    # A function that can be used by tfp.optimizer.lbfgs_minimize.
    @jit
    def f(params_1d):
        # convert the 1d parameters back to pytree format
        params = update(params_1d)
        # calculate gradients and convert to 1D tf.Tensor
        grads, loss_info = grad(lossf, has_aux=True)(params, data)
        # convert the grad to 1d arrays
        grads_1d = flat_utl.ravel_pytree(grads)[0]
        loss_value = loss_info[0]

        # # store loss value so we can retrieve later
        call(lambda x: f.loss.append(x), loss_info[0:4])
        call(lambda x: print(f"Step: NaN | Loss: {x[0]:.4e} | Loss_d: {x[1]:.4e}"
                  f" Loss_e: {x[2]:.4e} | Loss_b: {x[3]:.4e}"), loss_info)
        return loss_value, grads_1d

    # store these information as members so we can use them outside the scope
    f.update = update
    f.loss = []
    return f


# define the function to apply the L-BFGS optimizer
def lbfgs_optimizer(lossf, params, data, epoch):
    func_lbfgs = lbfgs_function(lossf, params, data)
    # convert initial model parameters to a 1D array
    init_params_1d = flat_utl.ravel_pytree(params)[0]
    # calculate the effective number of iteration
    max_nIter = jnp.int32(epoch / 3)
    # train the model with L-BFGS solver
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func_lbfgs, initial_position=init_params_1d,
        tolerance=1e-10, max_iterations=max_nIter)
    params = func_lbfgs.update(results.position)
    num_iter = results.num_objective_evaluations
    loss_all = func_lbfgs.loss
    print(f" Total iterations: {num_iter}")
    return params, loss_all
