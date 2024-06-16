'''
@author: Yongji Wang
Goal: "load_icedata.py" normalize the remote-sensing data of ice shelves,
and re-organize the data in a form that is required for the PINN training
'''

import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat
from pathlib import Path

# find the root directory
rootdir = Path(__file__).parent

def iceshelf_data(Datafile, step):
    '''
    :param n_col:      number of collocation point
    :return X_smp, U_smp, X_ct, n_ct, data_info
    '''

    #%% load data and unify the unit
    DataPath = str(rootdir.joinpath(Datafile))
    data = loadmat(DataPath)

    xall = data['x_1']   # unit [m] position
    yall = data['y_1']   # unit [m] position
    uall = data['u_1']   # unit [m/year] ice velocity
    vall = data['v_1']   # unit [m/year] ice velocity
    hall = data['H_1']   # unit [m] ice thickness
    sall = data['S_1']   # unit [m] ice thickness

    xout = xall[0::step, 0::step]
    yout = yall[0::step, 0::step]

    xct = data['xct']
    yct = data['yct']
    nnct = data['nnct']
    maskCrack = data['maskCrack']

    uall = uall / 365 / 24 / 3600  # change velocity unit to [m/s]
    vall = vall / 365 / 24 / 3600  # change velocity unit to [m/s]

    fullsize = xall.shape

    #%% flatten the data into 1d array

    x0 = xall.flatten()
    y0 = yall.flatten()
    u0 = uall.flatten()
    v0 = vall.flatten()
    h0 = hall.flatten()
    s0 = sall.flatten()
    isCrack0 = maskCrack.flatten()

    idxval = jnp.where(~np.isnan(x0))[0]
    x = x0[idxval, None]
    y = y0[idxval, None]
    u = u0[idxval, None]
    v = v0[idxval, None]
    h = h0[idxval, None]
    s = s0[idxval, None]
    isCrack = isCrack0[idxval, None]

    # find the indices in the array of the variables at crack position
    idx_notCrack = jnp.where(isCrack == 0)[0]

    #%% calculate the magnitude of each output variable for normalization later

    x_mean = jnp.mean(x)
    x_range = (x.max() - x.min()) / 2

    y_mean = jnp.mean(y)
    y_range = (y.max() - y.min()) / 2

    u_mean = jnp.mean(u)
    u_range = jnp.std(u) * 2

    v_mean = jnp.mean(v)
    v_range = jnp.std(v) * 2

    h_mean = jnp.mean(h)
    h_range = jnp.std(h) * 2

    s_mean = jnp.mean(s)
    s_range = jnp.std(s) * 2

    x_n = (x - x_mean) / x_range
    y_n = (y - y_mean) / y_range
    u_n = (u - u_mean) / u_range
    v_n = (v - v_mean) / v_range
    s_n = (s - s_mean) / s_range
    h_n = (h) / h_mean

    xct_n = (xct - x_mean) / x_range
    yct_n = (yct - y_mean) / y_range

    xo_n = (xout - x_mean) / x_range
    yo_n = (yout - y_mean) / y_range

    data_raw = [x0, y0, u0, v0, h0, s0]
    data_norm = [x_n, y_n, u_n, v_n, h_n, s_n]
    data_out = [xo_n, yo_n]

    # data_scale must be the shape of (1,5) or (5,), not (5,1)
    data_mean = jnp.hstack([x_mean, y_mean, u_mean, v_mean, h_mean, s_mean])
    data_range = jnp.hstack([x_range, y_range, u_range, v_range, h_range, s_range])

    # gathering all the data information
    data_info = [data_mean, data_range, data_norm, data_raw, data_out, idxval, fullsize]

    #%% generate the sampling points and collocation points

    # group the input and output into matrix
    X_star = jnp.hstack((x_n, y_n))
    X_ct = jnp.hstack((xct_n, yct_n))
    # sequence of output matrix column is u,v,h
    U_star = jnp.hstack((u_n, v_n, h_n))

    # obtain the training data not in the crack
    X_star = X_star[idx_notCrack, :]
    U_star = U_star[idx_notCrack, :]

    return X_star, U_star, X_ct, nnct, data_info


