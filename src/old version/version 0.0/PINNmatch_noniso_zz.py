import sys
import os
import argparse
import jax
import jax.config as config
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_map
from jax import random, jit, lax, grad, vmap, pmap
import jax.flatten_util as flat_utl
from jax.experimental.host_callback import call
from tensorflow_probability.substrates import jax as tfp
import time
import functools
from scipy.io import savemat
from pathlib import Path
from info_SSAeqn_nszz import vectgrad, gov_eqn, front_eqn
from info_data_multi import icedata


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
    return np.asarray(var_2d)


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
def ice_init_MLP(parent_key, n_hl, n_unit, ng=1):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''
    # set the neural network shape for u, v, h
    layers1 = [2] + n_hl * [n_unit] + [3]
    # set the neural network shape for mu
    layers2 = [2] + n_hl * [n_unit] + [2]

    # generate the random key for each network (in list format)
    _, *keys = random.split(parent_key, 2*ng+1)

    # generate weights and biases for each group
    params_u = tree_map(lambda x: init_MLP(x, layers1), keys[0:ng])
    params_mu = tree_map(lambda x: init_MLP(x, layers2), keys[ng:])

    return dict(net_u=params_u, net_mu=params_mu)


# wrapper to create solution function with given domain size
def ice_pred_create(scale, scl=1, act_s=0):
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
        # modify the gradient with the same scale
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


# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


# take the nth power root with original sign
def nthrt(x, n):
    return jnp.sign(x) * jnp.abs(x) ** (1/n)


def loss_create(predNN, gradNN, scale, idxgall, lw, loss_ref):
    ''' a function factory to create the loss function based on given info
    :param loss_ref: loss value at the initial of the training
    :return: a loss function (callable)
    '''

    # obtain the viscosity and strain rate scale in each sub-region
    all_info = jnp.array(tree_map(lambda x: sub_scale(scale[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.min(scale_info, axis=0)
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    uvh0 = jnp.hstack([u0, v0, h0])
    um, vm = jnp.split(mean_nm, 2, axis=1)

    def loss_sub(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)
        # load the data of normalization condition
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]
        # load the position and weight of collocation points
        x_col = data['col'][0][idx]
        x_bd = data['bd'][0][idx]
        nn_bd = data['bd'][1][idx]

        # calculate the output of network
        output = net(x_smp)
        u_pred = output[:, 0:3]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]

        # calculate the residue of equation
        f_pred = gov_eqn(net, x_col, scale[idx])[0]
        f_bd = front_eqn(net, x_bd, nn_bd, scale[idx])[0]

        # calculate the mean squared error of normalization cond.
        data_err = ms_error(u_pred - u_smp) * uvh0[idx]
        # calculate the mean squared error of equation
        eqn_err = ms_error(f_pred) * term0[idx]
        bd_err = ms_error(f_bd) * h0[idx]
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred)) / 2) * mu0[idx]

        # group all the error for output
        err_all = jnp.hstack([data_err, eqn_err, bd_err, sp_err])
        return err_all

    def loss_match(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)
        fgovterm = lambda x, id: gov_eqn(lambda x: net(x, id), x, scale[id])[1]
        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        # C0 stitching condition at the boundary
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx]
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx]
        h_md1 = (U_md1[:, 2:3]) * h0[idx]
        mu_md1 = (U_md1[:, 3:5]) * mu0[idx]
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2*jnp.log(mu_md1)])
        U_md2 = net(x_md[:, 2:4], idx+1)
        u_md2 = (U_md2[:, 0:1] + um[idx+1]) * u0[idx+1]
        v_md2 = (U_md2[:, 1:2] + vm[idx+1]) * v0[idx+1]
        h_md2 = (U_md2[:, 2:3]) * h0[idx+1]
        mu_md2 = (U_md2[:, 3:5]) * mu0[idx+1]
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2*jnp.log(mu_md2)])
        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)

        # c1 stitching condition at the boundary
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx]
        dh_md1 = dU_md1[:, 4:6] * dh0[idx]
        dvars_md1 = jnp.hstack([duv_md1, dh_md1])
        dU_md2 = gdnet(x_md[:, 2:4], idx+1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx+1]
        dh_md2 = dU_md2[:, 4:6] * dh0[idx+1]
        dvars_md2 = jnp.hstack([duv_md2, dh_md2])
        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        # c2 stitching condition at the boundary
        # calculate the residue of equation
        term_md1 = fgovterm(x_md[:, 0:2], idx)[:, 0:-1] * term0[idx]
        term_md2 = fgovterm(x_md[:, 2:4], idx+1)[:, 0:-1] * term0[idx+1]
        # calculate the c2 error
        match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitching conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.5])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        reg_err_list = tree_map(lambda x: loss_sub(params, data, x), idxgall)
        reg_err = jnp.mean(jnp.array(reg_err_list), axis=0)
        # calculate the error at the matching boundary
        match_err_list = tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1])
        match_err = jnp.mean(jnp.array(match_err_list), axis=0)
        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        data_w = jnp.array([1., 1., 0.6])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        sp_w = jnp.array([1.])
        md_w = jnp.ones(match_err.shape[0])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, sp_w, md_w])

        loss_each = err_all * wgh_all
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(loss_each[0:3])
        loss_eqn = jnp.sum(loss_each[3:5])
        loss_bd = jnp.sum(loss_each[5:7])
        loss_sp = jnp.sum(loss_each[7:8])
        loss_md = jnp.sum(loss_each[8:])

        # loading the pre-saved loss parameter
        lw = loss_fun.lw
        lref = loss_fun.ref
        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + lw[2] * loss_md + lw[3] * loss_sp)
        loss_n = loss / lref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md, loss_sp]), err_all])
        return loss_n, loss_info

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
    nc = jnp.int32(10000)
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
            f" Loss_e: {loss_info[2]:.4e} | Loss_b: {loss_info[3]:.4e} | Loss_m: {loss_info[4]:.4e}", file=sys.stderr)

        # saving the loss 
        loss_all.append(loss_info[0:])

    # obtain the total loss in the last iterations
    lossend = jnp.array(loss_all[-nc:])[:, 0]
    # find the minimum loss value
    lmin = jnp.mean(lossend)
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
                  f" Loss_e: {x[2]:.4e} | Loss_b: {x[3]:.4e} | Loss_m: {x[4]:.4e}"), loss_info)
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


def data_func_create(data_all, idxgall, n_pt):
    # obtain the number of sub-group
    ng = len(idxgall)
    # load the data within ice shelf
    X_star = tree_map(lambda x: data_all[x][0], idxgall)
    U_star = tree_map(lambda x: data_all[x][1], idxgall)
    # load the data at the ice front
    X_ct = tree_map(lambda x: data_all[x][2], idxgall)
    nn_ct = tree_map(lambda x: data_all[x][3], idxgall)
    # load the data at the sub-region boundary
    Xraw_md = tree_map(lambda x: data_all[x][-1], idxgall)
    X_md = Xraw_md[0:-1]
    n_md = [jnp.array(1.)] * (ng-1)
    # load the data at the connect
    for l in range(ng - 1):
        # obtain the boundary from the previous subregion
        if l == 0:
            X_md1 = Xraw_md[l]
        else:
            n_md0 = n_md[l - 1]
            X_md1 = Xraw_md[l][n_md0:]
        # obtain the boundary from the next subregion
        n_md1 = X_md1.shape[0]
        X_md2 = Xraw_md[l + 1][0:n_md1, :]
        # pair the boundary in both sub-regions
        X_mdp = jnp.hstack([X_md1, X_md2])
        n_md[l] = n_md1
        X_md[l] = X_mdp

    # create the index of data points within the domain and boundary
    idx_data = tree_map(lambda x: jnp.arange(X_star[x].shape[0]), idxgall)
    idx_bd = tree_map(lambda x: jnp.arange(X_ct[x].shape[0]), idxgall)
    idx_md = tree_map(lambda x: jnp.arange(X_md[x].shape[0]), idxgall[0:-1])

    # define the function that can re-sampling for each calling
    def dataf(key):
        # generate the new random key
        _, *keys = random.split(key, 4*ng)

        # sampling the data point based on the index
        idx_smp = tree_map(lambda x, y: random.choice(x, y, [n_pt[0]], replace=False), keys[0:ng], idx_data)
        X_smp = tree_map(lambda x, y: X_star[x][y], idxgall, idx_smp)
        U_smp = tree_map(lambda x, y: U_star[x][y], idxgall, idx_smp)

        # generate a random sample of collocation point within the domain
        idx_col = tree_map(lambda x, y: random.choice(x, y, [n_pt[1]], replace=False), keys[ng:(2*ng)], idx_data)
        # sampling the data point based on the index
        X_col = tree_map(lambda x, y: X_star[x][y], idxgall, idx_col)

        # generate a random index of the data at ice front
        idx_cbd = tree_map(lambda x, y: random.choice(x, y, [n_pt[2]], replace=False), keys[(2*ng):(3*ng)], idx_bd)
        # sampling the data point based on the index
        X_bd = tree_map(lambda x, y: X_ct[x][y], idxgall, idx_cbd)
        nn_bd = tree_map(lambda x, y: nn_ct[x][y], idxgall, idx_cbd)

        # generate a random index of the data at matching boundary
        idx_mbd = tree_map(lambda x, y: random.choice(x, y, [n_pt[3]], replace=False), keys[(3*ng):(4*ng-1)], idx_md)
        # sampling the data point based on the index
        X_mbd = tree_map(lambda x, y: X_md[x][y], idxgall[0:-1], idx_mbd)

        # group all the data and collocation points
        data = dict(smp=[X_smp, U_smp], col=[X_col],  bd=[X_bd, nn_bd], md=[X_mbd])
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
    um, vm, h0 = dmean[2:5]
    # find the maximum velocity and length scale
    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    # calculate the scale of viscosity and strain rate
    mu0 = rho * gd * h0 * (l0m / u0m)
    du0 = u0m / l0m
    dh0 = h0 / l0m
    term0 = h0**2 / l0m
    return u0, v0, h0, mu0, du0, dh0, term0, um/u0, vm/v0


def predict(params, data_out, scale, idx, nsp=4):
    xout, yout = data_out[idx]
    xo = xout.flatten()[:, None]
    yo = yout.flatten()[:, None]
    idxval = jnp.where(~jnp.isnan(xo))[0]
    x_pred = jnp.hstack([xo, yo])[idxval]

    # calculate the solution and equation function
    f_u = lambda x: pred_u(params, x, idx)
    f_gu = lambda x: grad_u(params, x, idx)
    f_eqn = lambda x: gov_eqn(f_u, x, scale[idx])

    # separate the array for the memory limit
    x_psp = jnp.array_split(x_pred, nsp)
    idxsp = jnp.arange(nsp).tolist()
    # initialize the list to save the sub-group data
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
    return x_pred, uvhm, duvh, eqn, term, idxval


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
    lw = [0.05, 0.1, 1, 0.1]

    # number of sampling points
    n_smp = 6000
    n_col = 6000
    n_cbd = 300
    n_mbd = 1000
    n_pt = jnp.array([n_smp, n_col, n_cbd, n_mbd], dtype='int32')
    n_pt2 = jnp.array(n_pt*2, dtype='int32')  # increase the points for L-BFGS

    # choose the shelf data
    args = parser.parse_args()
    shelf_name = args.ice_shelf
    # create the dataset filename
    DataFile = 'Ice_' + shelf_name + '_data_match.mat'
    # check whether sub-folder exists
    outdir = rootdir.joinpath(shelf_name)
    isExist = os.path.exists(outdir)
    # create the sub-folder if not exist
    if isExist is False:
        os.mkdir(outdir)

    # obtain the data for training
    data_all, idxgall = icedata(DataFile, step=1)
    scale = tree_map(lambda x: data_all[x][4][0:2], idxgall)

    # initialize the weights and biases of the network
    trained_params = ice_init_MLP(keys[0], n_hl, n_unit, ng=len(idxgall))

    # create the solution function
    pred_u, grad_u = ice_pred_create(scale, scl=1, act_s=0)

    # create the data function for Adam
    dataf = data_func_create(data_all, idxgall, n_pt)
    keys_adam = random.split(keys[1], 5)
    data = dataf(keys_adam[0])
    # create the data function for L-BFGS
    dataf_l = data_func_create(data_all, idxgall, n_pt2)
    key_lbfgs = keys[2]

    # calculate the loss function
    NN_loss = loss_create(pred_u, grad_u, scale, idxgall, lw, loss_ref=1)
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

    for l in range(9):
        loss1 = []
        loss2 = []
        if l < 4:
            # training with Adam
            trained_params, loss1 = adam_optimizer(keys_adam[l], NN_loss, trained_params, dataf, epoch1, lr=lr)
            NN_loss.lw[3] /= 2.4
        elif l >= 4:
            for k in range(2):
                # update the random key
                key_lbfgs = random.split(key_lbfgs, 1)[0]
                # resample the data
                data_l = dataf_l(key_lbfgs)
                # training with L-bfgs
                trained_params, loss_l = lbfgs_optimizer(NN_loss, trained_params, data_l, epoch2)
                loss2 = loss2 + loss_l

        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed, file=sys.stderr)

        # generate the last loss
        loss_all = np.array(loss1 + loss2)

        #%% calculate the output data

        idxval = tree_map(lambda x: data_all[x][4][-2], idxgall)
        dsize = tree_map(lambda x: data_all[x][4][-1], idxgall)
        data_norm = tree_map(lambda x: data_all[x][4][2], idxgall)

        data_out = tree_map(lambda x: data_all[x][4][4], idxgall)
        dsout = tree_map(lambda x: data_out[x][0].shape, idxgall)
        output = tree_map(lambda x: predict(trained_params, data_out, scale, x), idxgall)
        idxout = tree_map(lambda x: output[x][-1], idxgall)

        # convert to 2D data
        x = np.array(tree_map(lambda x: dataArrange(data_norm[x][0], idxval[x], dsize[x]), idxgall), dtype=object)
        y = np.array(tree_map(lambda x: dataArrange(data_norm[x][1], idxval[x], dsize[x]), idxgall), dtype=object)
        u_data = np.array(tree_map(lambda x: dataArrange(data_norm[x][2], idxval[x], dsize[x]), idxgall), dtype=object)
        v_data = np.array(tree_map(lambda x: dataArrange(data_norm[x][3], idxval[x], dsize[x]), idxgall), dtype=object)
        h_data = np.array(tree_map(lambda x: dataArrange(data_norm[x][4], idxval[x], dsize[x]), idxgall), dtype=object)

        # convert to 2D NN prediction
        u_p = np.array(tree_map(lambda x: dataArrange(output[x][1][:, 0:1], idxout[x], dsout[x]), idxgall), dtype=object)
        v_p = np.array(tree_map(lambda x: dataArrange(output[x][1][:, 1:2], idxout[x], dsout[x]), idxgall), dtype=object)
        h_p = np.array(tree_map(lambda x: dataArrange(output[x][1][:, 2:3], idxout[x], dsout[x]), idxgall), dtype=object)
        mu_p = np.array(tree_map(lambda x: dataArrange(output[x][1][:, 3:4], idxout[x], dsout[x]), idxgall), dtype=object)
        eta_p = np.array(tree_map(lambda x: dataArrange(output[x][1][:, 4:5], idxout[x], dsout[x]), idxgall), dtype=object)

        # convert to 2D derivative of prediction
        ux_p = np.array(tree_map(lambda x: dataArrange(output[x][2][:, 0:1], idxout[x], dsout[x]), idxgall), dtype=object)
        uy_p = np.array(tree_map(lambda x: dataArrange(output[x][2][:, 1:2], idxout[x], dsout[x]), idxgall), dtype=object)
        vx_p = np.array(tree_map(lambda x: dataArrange(output[x][2][:, 2:3], idxout[x], dsout[x]), idxgall), dtype=object)
        vy_p = np.array(tree_map(lambda x: dataArrange(output[x][2][:, 3:4], idxout[x], dsout[x]), idxgall), dtype=object)
        hx_p = np.array(tree_map(lambda x: dataArrange(output[x][2][:, 4:5], idxout[x], dsout[x]), idxgall), dtype=object)
        hy_p = np.array(tree_map(lambda x: dataArrange(output[x][2][:, 5:6], idxout[x], dsout[x]), idxgall), dtype=object)
        # str_p = tree_map(lambda x: dataArrange(output[x][2][:, 6:7], idxout[x], dsout[x]), idxgall)

        # convert to 2D equation residue
        e1 = np.array(tree_map(lambda x: dataArrange(output[x][3][:, 0:1], idxout[x], dsout[x]), idxgall), dtype=object)
        e2 = np.array(tree_map(lambda x: dataArrange(output[x][3][:, 1:2], idxout[x], dsout[x]), idxgall), dtype=object)

        # convert to 2D equation term value
        e11 = np.array(tree_map(lambda x: dataArrange(output[x][4][:, 0:1], idxout[x], dsout[x]), idxgall), dtype=object)
        e12 = np.array(tree_map(lambda x: dataArrange(output[x][4][:, 1:2], idxout[x], dsout[x]), idxgall), dtype=object)
        e13 = np.array(tree_map(lambda x: dataArrange(output[x][4][:, 2:3], idxout[x], dsout[x]), idxgall), dtype=object)
        e21 = np.array(tree_map(lambda x: dataArrange(output[x][4][:, 3:4], idxout[x], dsout[x]), idxgall), dtype=object)
        e22 = np.array(tree_map(lambda x: dataArrange(output[x][4][:, 4:5], idxout[x], dsout[x]), idxgall), dtype=object)
        e23 = np.array(tree_map(lambda x: dataArrange(output[x][4][:, 5:6], idxout[x], dsout[x]), idxgall), dtype=object)
        strate = np.array(tree_map(lambda x: dataArrange(output[x][4][:, -1:], idxout[x], dsout[x]), idxgall), dtype=object)

        # saving the scale of mu0
        dmean = np.array(tree_map(lambda x: scale[x][0], idxgall))
        drange = np.array(tree_map(lambda x: scale[x][1], idxgall))
        mu0 = np.array(tree_map(lambda x: sub_scale(scale[x]), idxgall))

        params_u = trained_params['net_u']
        params_mu = trained_params['net_mu']
        params_u = tree_map(lambda z: np.array(z), params_u)
        params_mu = tree_map(lambda z: np.array(z), params_mu)
        pm0_u = tree_map(lambda z: np.array(params_u[z], dtype=object), idxgall)
        pm0_mu = tree_map(lambda z: np.array(params_mu[z], dtype=object), idxgall)
        pm_u = np.array(pm0_u, dtype=object)
        pm_mu = np.array(pm0_mu, dtype=object)

        #%% saving the output

        mdic = {"x": x, "y": y, "u": u_p, "v": v_p, "h": h_p, "u_g": u_data, "v_g": v_data, "h_g": h_data,
                "u_x": ux_p, "u_y": uy_p, "v_x": vx_p, "v_y": vy_p, "h_x": hx_p, "h_y": hy_p, "str": strate,
                "e11": e11, "e12": e12, "e13": e13, "e21": e21, "e22": e22, "e23": e23, "e1": e1, "e2": e2,
                "mu": mu_p, "eta": eta_p, "params_u": pm_u, "params_mu": pm_mu, "loss": loss_all,
                "v_mean": dmean, "v_range": drange, "mu0": mu0}
        FileName = 'Ice_' + shelf_name + '_nszz_match_idx%3d_iter%1d.mat' % (seed, l)
        FilePath = str(outdir.joinpath(FileName))
        savemat(FilePath, mdic)