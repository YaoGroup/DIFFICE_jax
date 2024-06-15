import jax.numpy as jnp
from jax import random


# wrapper to create function that can re-sample the dataset and collocation points
def data_sample_create(data_all, n_pt):
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

