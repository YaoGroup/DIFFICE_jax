'''
@author: Yongji Wang
Goal: "load_syndata.py" reorganize the synthetic data into a new format
that can be directly loaded into the PINN codes for the training
'''

import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat
from pathlib import Path

# find the root directory
rootdir = Path(__file__).parent

def iceshelf_data(filename, step):
    '''
    :param filename: name of the data file
    :return X_smp, U_smp, X_ct, n_ct, data_info
    '''

    #%% load the data file
    DataPath = str(rootdir.joinpath(filename))
    data = loadmat(DataPath)

    # extract each variable
    xall = data['xq']   # unit [m] position
    yall = data['yq']   # unit [m] position
    uall = data['uq']   # unit [m/s] ice velocity
    vall = data['vq']   # unit [m/s] ice velocity
    hall = data['hq']   # unit [m] ice thickness
    # record the full size of the input data
    fullsize = xall.shape

    # set the position of the calving front (right side of the domain)
    xct = xall[:, -3:].flatten()[:, None]
    yct = yall[:, -3:].flatten()[:, None]
    # set the unit normal vector of the calving front (towards right)
    nnct = jnp.hstack([jnp.ones(xct.shape), jnp.zeros(xct.shape)])

    # resolution reduction for output dataset if necessary (for low memory usage)
    xout = xall[0::step, 0::step]
    yout = yall[0::step, 0::step]

    #%% flatten the data into 1d array
    x0 = xall.flatten()
    y0 = yall.flatten()
    u0 = uall.flatten()
    v0 = vall.flatten()
    h0 = hall.flatten()

    idxval = jnp.where(~np.isnan(x0) & (~np.isnan(u0)))[0]
    x = x0[idxval, None]
    y = y0[idxval, None]
    u = u0[idxval, None]
    v = v0[idxval, None]
    h = h0[idxval, None]

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

    x_n = (x - x_mean) / x_range
    y_n = (y - y_mean) / y_range
    u_n = (u - u_mean) / u_range
    v_n = (v - v_mean) / v_range
    h_n = (h) / h_mean

    xct_n = (xct - x_mean) / x_range
    yct_n = (yct - y_mean) / y_range

    xo_n = (xout - x_mean) / x_range
    yo_n = (yout - y_mean) / y_range

    data_raw = [x0, y0, u0, v0, h0]
    data_norm = [x_n, y_n, u_n, v_n, h_n]
    data_out = [xo_n, yo_n]

    # data_scale must be the shape of (1,5) or (5,), not (5,1)
    data_mean = jnp.hstack([x_mean, y_mean, u_mean, v_mean, h_mean])
    data_range = jnp.hstack([x_range, y_range, u_range, v_range, h_range])

    # gathering all the data information
    data_info = [data_mean, data_range, data_norm, data_raw, data_out, idxval, fullsize]

    #%% generate the sampling points and collocation points

    # group the input and output into matrix
    X_star = jnp.hstack((x_n, y_n))
    X_ct = jnp.hstack((xct_n, yct_n))
    # sequence of output matrix column is u,v,h
    U_star = jnp.hstack((u_n, v_n, h_n))

    return X_star, U_star, X_ct, nnct, data_info


