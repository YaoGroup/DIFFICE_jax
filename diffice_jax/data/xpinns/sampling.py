import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map


def data_sample_create(data_all, idxgall, n_pt):
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

    # create the index of velocity data points within all sub-regions
    idx_data = tree_map(lambda x: jnp.arange(X_star[x][0].shape[0]), idxgall)
    # create the index of thickness data points within all sub-regions
    idxh_data = tree_map(lambda x: jnp.arange(X_star[x][1].shape[0]), idxgall)
    # create the index of data points for all sub-regions at the calving front
    idx_bd = tree_map(lambda x: jnp.arange(X_ct[x].shape[0]), idxgall)
    # create the index of data points at the interface between different pairs of sub-regions
    idx_md = tree_map(lambda x: jnp.arange(X_md[x].shape[0]), idxgall[0:-1])

    # define the function that can re-sampling for each calling
    def dataf(key):
        # generate the new random key
        _, *keys = random.split(key, 4*ng)

        # sampling the velocity data point based on the index
        idx_smp = tree_map(lambda x, y: random.choice(x, y, [n_pt[0]], replace=False), keys[0:ng], idx_data)
        X_smp = tree_map(lambda x, y: X_star[x][0][y], idxgall, idx_smp)
        U_smp = tree_map(lambda x, y: U_star[x][0][y], idxgall, idx_smp)

        # sampling the thickness data point based on the index
        idxh_smp = tree_map(lambda x, y: random.choice(x, y, [n_pt[1]], replace=False), keys[0:ng], idxh_data)
        Xh_smp = tree_map(lambda x, y: X_star[x][1][y], idxgall, idxh_smp)
        H_smp = tree_map(lambda x, y: U_star[x][1][y], idxgall, idxh_smp)

        # generate a random sample of collocation point within the domain
        idx_col = tree_map(lambda x, y: random.choice(x, y, [n_pt[2]], replace=False), keys[ng:(2*ng)], idx_data)
        # sampling the collocation point based on the position of velocity data
        X_col = tree_map(lambda x, y: X_star[x][0][y], idxgall, idx_col)

        # generate a random index of the data at ice front
        idx_cbd = tree_map(lambda x, y: random.choice(x, y, [n_pt[3]], replace=False), keys[(2*ng):(3*ng)], idx_bd)
        # sampling the data point based on the index
        X_bd = tree_map(lambda x, y: X_ct[x][y], idxgall, idx_cbd)
        nn_bd = tree_map(lambda x, y: nn_ct[x][y], idxgall, idx_cbd)

        # generate a random index of the data at matching boundary
        idx_mbd = tree_map(lambda x, y: random.choice(x, y, [n_pt[4]], replace=False), keys[(3*ng):(4*ng-1)], idx_md)
        # sampling the data point based on the index
        X_mbd = tree_map(lambda x, y: X_md[x][y], idxgall[0:-1], idx_mbd)

        # group all the data and collocation points
        data = dict(smp=[X_smp, U_smp, Xh_smp, H_smp], col=[X_col],  bd=[X_bd, nn_bd], md=[X_mbd])
        return data
    return dataf


