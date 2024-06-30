import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
import jax.config as config
import jax.numpy as jnp
import numpy as np
import optax
from jax import random, jit, lax, grad, vmap, pmap
from jax.tree_util import tree_map
import jax.flatten_util as flat_utl
from jax.experimental.host_callback import call
from tensorflow_probability.substrates import jax as tfp
from pyDOE import lhs
import time
import functools
from scipy.io import savemat
from pathlib import Path
from info_SSAeqn_nsxy import vectgrad, gov_eqn, front_eqn
from info_data import icesheet_data


# find the root directory
rootdir = Path(__file__).parent
# rootdir = Path('/scratch/gpfs/yw1705')
# rootdir = Path('/scratch/users/yongjiw')

# change JAX to double precision
config.update('jax_enable_x64', True)


def dataArrange(var, idxval, dsize):
    nanmat = jnp.empty(dsize)
    nanmat = nanmat.at[:].set(np.NaN)
    var_1d = nanmat.flatten()[:, None]
    var_1d = var_1d.at[idxval].set(var)
    var_2d = jnp.reshape(var_1d, dsize)
    return var_2d


# initialize the neural network weights and biases
def init_MLP(parent_key, layer_widths):
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


# generate weights and biases for all variables of CLM problem
def ice_init_MLP(parent_key, n_hl, n_unit):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''
    # set the neural network shape for u, v, h
    layers1 = [2] + n_hl * [n_unit] + [3]
    # set the neural network shape for mu and eta
    layers2 = [2] + n_hl * [n_unit] + [2]

    # generate the random key for each network
    keys = random.split(parent_key, 2)
    # generate weights and biases for
    params_u = init_MLP(keys[0], layers1)
    params_mu = init_MLP(keys[0], layers2)
    return [params_u, params_mu]


# wrapper to create solution function with given domain size
def ice_pred_create(scale, scl=1, act_s=0):
    '''
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    '''
    def f(params, x):
        # generate the NN
        uvh = neural_net(params[0], x, scl, act_s)
        mu = neural_net(params[1], x, scl, act_s)
        sol = jnp.hstack([uvh, jnp.exp(mu)])
        return sol

    def gradf(params, x):
        drange = scale[1]
        lx0, ly0, u0, v0 = drange[0:4]
        u0m = lax.max(u0, v0)
        l0m = lax.max(lx0, ly0)
        ru0 = u0 / u0m
        rv0 = v0 / u0m
        rx0 = lx0 / l0m
        ry0 = ly0 / l0m
        coeff = jnp.hstack([ru0/rx0, ru0/ry0, rv0/rx0, rv0/ry0, 1/rx0, 1/ry0])
        # load the network
        net = lambda x: f(params, x)
        # calculate the gradient
        grad = vectgrad(net, x)[0]
        # modify the gradient with the same scale
        duvh = grad[:, 0:6] * coeff
        return duvh

    return f, gradf


# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


def loss_create(predNN, scale, lw, loss_ref):
    ''' a function factory to create the loss function based on given info
    :param loss_ref: loss value at the initial of the training
    :return: a loss function (callable)
    '''

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z)
        # load the data of normalization condition
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        output = net(x_smp)
        u_pred = output[:, 0:3]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]

        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_err = ms_error(u_pred - u_smp)
        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        bd_err = ms_error(f_bd)
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred))/2)

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)
        loss_sp = jnp.sum(sp_err)

        # loading the pre-saved loss parameter
        lw = loss_fun.lw
        lref = loss_fun.ref
        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + lw[2] * loss_sp) / lref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_sp]),
                                data_err, eqn_err, bd_err])
        return loss, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lw = lw
    loss_fun.ref = loss_ref
    return loss_fun


# create the Adam minimizer
@functools.partial(jit, static_argnames=("lossf", "opt"))
def adam_minimizer(lossf, params, data, opt, opt_state):
    """Basic gradient update step based on the opt optimizer."""
    grads, loss_info = grad(lossf, has_aux=True)(params, data)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss_info, opt_state


def adam_optimizer(key, lossf, params, dataf, epoch, lr=1e-3):
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
        if step % 100 == 0 and step > 0:
            # print the results
            print(f"Step: {step} | Loss: {loss_info[0]:.4e} | Loss_d: {loss_info[1]:.4e} |"
            f" Loss_e: {loss_info[2]:.4e} | Loss_b: {loss_info[3]:.4e}", file=sys.stderr)

        # saving the loss
        loss_all.append(loss_info[0:])

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
        loss_all.append(loss_info[0:])

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
        call(lambda x: f.loss.append(x), loss_info[0:], result_shape=None)
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
    # history = func_lbfgs.loss
    num_iter = results.num_objective_evaluations
    loss_all = func_lbfgs.loss
    print(f" Total iterations: {num_iter}")
    return params, loss_all



def data_func_create(data_all, n_pt):
    # load the data within ice shelf
    X_star = data_all[0]
    U_star = data_all[1]
    # load the data at the ice front
    X_ct = data_all[2]
    nn_ct = data_all[3]

    # obtain the number of data points and points at the boundary
    n_data = X_star.shape[0]
    n_bd = X_ct.shape[0]

    # define the function that can re-sampling for each calling
    def dataf(key):
        # generate the new random key
        keys = random.split(key, 3)

        # sampling the data point based on the index
        idx_smp = random.choice(keys[0], jnp.arange(n_data), [n_pt[0]])
        X_smp = X_star[idx_smp]
        U_smp = U_star[idx_smp]

        # generate a random sample of collocation point within the domain
        idx_col = random.choice(keys[1], jnp.arange(n_data), [n_pt[1]])
        # sampling the data point based on the index
        X_col = X_star[idx_col]

        # generate a random index of the data at ice front
        idx_cbd = random.choice(keys[2], jnp.arange(n_bd), [n_pt[2]])
        # sampling the data point based on the index
        X_bd = X_ct[idx_cbd]
        nn_bd = nn_ct[idx_cbd]

        # group all the data and collocation points
        data = dict(smp=[X_smp, U_smp], col=[X_col],  bd=[X_bd, nn_bd])
        return data
    return dataf


def sub_scale(scale):
    # define the global parameter
    rho = 917
    rho_w = 1030
    gd = 9.8 * (1 - rho / rho_w)  # gravitational acceleration
    # load the scale information
    dmean, drange = scale
    lx0, ly0, u0, v0 = drange[0:4]
    h0 = dmean[4]
    # find the maximum velocity and length scale
    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    # calculate the scale of viscosity and strain rate
    mu0 = rho * gd * h0 * (l0m / u0m)
    str0 = u0m/l0m
    return mu0, str0


"""Set the conditions of the problem"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ice_shelf", type=str)

    # select the random seed
    seed = np.random.choice(3000, 1)[0]
    key = random.PRNGKey(seed)
    np.random.seed(seed)

    # create the subkeys
    keys = random.split(key, 4)

    # select the size of neural network
    n_hl = 6
    n_unit = 40
    lw = [0.05, 0.1, 0.25]

    # number of sampling points
    n_smp = 6000
    n_col = 6000
    n_cbd = 500
    n_pt = jnp.array([n_smp, n_col, n_cbd], dtype='int32')
    n_pt2 = n_pt * 2

    # choose the shelf data
    args = parser.parse_args()
    shelf_name = args.ice_shelf
    # create the dataset filename
    DataFile = 'Ice_' + shelf_name + '_data_all.mat'
    # check whether sub-folder exists
    outdir = rootdir.joinpath(shelf_name)
    isExist = os.path.exists(outdir)
    # create the sub-folder if not exist
    if isExist is False:
        os.mkdir(outdir)

    # obtain the data for training
    data_all = icesheet_data(DataFile, step=1)
    scale = data_all[4][0:2]

    # initialize the weights and biases of the network
    trained_params = ice_init_MLP(keys[0], n_hl, n_unit)

    # create the solution function
    pred_u, grad_u = ice_pred_create(scale, scl=1, act_s=0)

    # create the data function for Adam
    dataf = data_func_create(data_all, n_pt)
    keys_adam = random.split(keys[1], 5)
    data = dataf(keys_adam[0])
    # create the data function for L-BFGS
    dataf_l = data_func_create(data_all, n_pt2)
    key_lbfgs = keys[2]

    # calculate the loss function
    NN_loss = loss_create(pred_u, scale, lw, loss_ref=1)
    # update the loss reference based on the real loss
    NN_loss.ref = NN_loss(trained_params, data)[0]

    # set the training iteration
    epoch1 = 200000
    epoch2 = 100000

    """Training using Adam"""

    # set the learning rate for Adam
    lr = 1e-3
    # training the neural network
    start_time = time.time()

    for l in range(10):
        loss1 = []
        loss2 = []
        if l < 5:
            # training with Adam
            trained_params, loss1 = adam_optimizer(keys_adam[l], NN_loss, trained_params, dataf, epoch1, lr=lr)
            NN_loss.lw[2] /= 2.4
        elif l >= 5:
            for k in range(2):
                # update the random key
                key_lbfgs = random.split(key_lbfgs, 1)[0]
                # resample the data
                data_l = dataf_l(key_lbfgs)
                # training with L-bfgs
                trained_params, loss_l = lbfgs_optimizer(NN_loss, trained_params, data_l, epoch2)
                loss2 = loss2 + loss_l

        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

        # generate the last loss
        loss_all = jnp.array(loss1+loss2)

        #%% calculate the output data

        idxval, dsize = data_all[4][-2:]
        data_norm = data_all[4][2]
        x_star, y_star, u_star, v_star, h_star = data_norm[0:5]
        xout, yout = data_all[4][4][0:2]
        dsout = xout.shape
        xo = xout.flatten()[:, None]
        yo = yout.flatten()[:, None]
        idxout = jnp.where(~jnp.isnan(xo))[0]
        x_pred = jnp.hstack([xo, yo])[idxout]

        # calculate the solution and equation function
        f_u = lambda x: pred_u(trained_params, x)
        f_gu = lambda x: grad_u(trained_params, x)
        f_eqn = lambda x: gov_eqn(f_u, x, scale)

        # separate the array for the memory limit
        nsp = 4
        # separate the array for the memory limit
        x_psp = jnp.array_split(x_pred, nsp)
        idxsp = jnp.arange(nsp).tolist()
        # calculate the data for each sub-group
        u_list = tree_map(lambda x: f_u(x_psp[x]), idxsp)
        du_list = tree_map(lambda x: f_gu(x_psp[x]), idxsp)
        eqnterm_list = tree_map(lambda x: f_eqn(x_psp[x]), idxsp)
        eqn_list = tree_map(lambda x: eqnterm_list[x][0], idxsp)
        term_list = tree_map(lambda x: eqnterm_list[x][1], idxsp)
        # combine the sub-group list into a long array
        uvhm = jnp.vstack(u_list)
        duvh = jnp.vstack(du_list)
        eqn = jnp.vstack(eqn_list)
        term = jnp.vstack(term_list)

        # convert to 2D data
        x = dataArrange(x_star, idxval, dsize)
        y = dataArrange(y_star, idxval, dsize)
        u_data = dataArrange(u_star, idxval, dsize)
        v_data = dataArrange(v_star, idxval, dsize)
        h_data = dataArrange(h_star, idxval, dsize)

        # convert to 2D NN prediction
        u_p = dataArrange(uvhm[:, 0:1], idxout, dsout)
        v_p = dataArrange(uvhm[:, 1:2], idxout, dsout)
        h_p = dataArrange(uvhm[:, 2:3], idxout, dsout)
        mu_p = dataArrange(uvhm[:, 3:4], idxout, dsout)
        eta_p = dataArrange(uvhm[:, 4:5], idxout, dsout)

        # convert to 2D derivative of prediction
        ux_p = dataArrange(duvh[:, 0:1], idxout, dsout)
        uy_p = dataArrange(duvh[:, 1:2], idxout, dsout)
        vx_p = dataArrange(duvh[:, 2:3], idxout, dsout)
        vy_p = dataArrange(duvh[:, 3:4], idxout, dsout)
        hx_p = dataArrange(duvh[:, 4:5], idxout, dsout)
        hy_p = dataArrange(duvh[:, 5:6], idxout, dsout)

        # convert to 2D equation residue
        e1 = dataArrange(eqn[:, 0:1], idxout, dsout)
        e2 = dataArrange(eqn[:, 1:2], idxout, dsout)

        # convert to 2D equation term value
        e11 = dataArrange(term[:, 0:1], idxout, dsout)
        e12 = dataArrange(term[:, 1:2], idxout, dsout)
        e13 = dataArrange(term[:, 2:3], idxout, dsout)
        e21 = dataArrange(term[:, 3:4], idxout, dsout)
        e22 = dataArrange(term[:, 4:5], idxout, dsout)
        e23 = dataArrange(term[:, 5:6], idxout, dsout)
        strate = dataArrange(term[:, -1:], idxout, dsout)

        # saving the scale of mu0 and str0
        dmean, drange = scale
        mu0 = sub_scale(scale)

        params = tree_map(lambda z: np.array(z), trained_params)
        pm = np.array(params, dtype=object)

        # %% saving the output

        mdic = {"x": np.asarray(x), "y": np.asarray(y), "u": np.asarray(u_p), "v": np.asarray(v_p), "h": np.asarray(h_p),
                "u_g": np.asarray(u_data), "v_g": np.asarray(v_data), "h_g": np.asarray(h_data),
                "u_x": np.asarray(ux_p), "u_y": np.asarray(uy_p), "v_x": np.asarray(vx_p), "v_y": np.asarray(vy_p),
                "h_x": np.asarray(hx_p), "h_y": np.asarray(hy_p), "str": np.asarray(strate),
                "e11": np.asarray(e11), "e12": np.asarray(e12), "e13": np.asarray(e13), "e21": np.asarray(e21),
                "e22": np.asarray(e22), "e23": np.asarray(e23), "e1": np.asarray(e1), "e2": np.asarray(e2),
                "mu": np.asarray(mu_p), "eta": np.asarray(eta_p), "params": pm, "loss": np.asarray(loss_all),
                "v_mean": np.asarray(dmean), "v_range": np.asarray(drange), "mu0": np.asarray(mu0)}
        FileName = 'Ice_' + shelf_name + '_nsxy_SGD_idx%3d_iter%1d.mat' % (seed, l)
        FilePath = str(outdir.joinpath(FileName))
        savemat(FilePath, mdic)