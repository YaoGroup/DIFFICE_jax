import sys
import os
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax import random, lax
import time
from scipy.io import savemat
from pathlib import Path
import pickle

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data.xpinns.preprocessing import normalize_data
from data.xpinns.sampling import data_sample_create
from equation.ssa_eqn_aniso_zz import vectgrad, gov_eqn, front_eqn
from model.xpinns.initialization import init_xpinns
from model.xpinns.networks import solu_create
from model.xpinns.loss import loss_aniso_create
from model.xpinns.prediction import predict
from optimizer.optimization import adam_optimizer, lbfgs_optimizer

# find the root directory
rootdir = Path(__file__).parent

#%% setting hyper-parameters

# select the random seed
seed = np.random.choice(3000, 1)[0]
key = random.PRNGKey(seed)
np.random.seed(seed)

# create the subkeys
keys = random.split(key, 4)

# select the size of neural network
n_hl = 6
n_unit = 40
# set the weights for 1. equation loss, 2. boundary condition loss
# 3. matching condition loss and 4. regularization loss
lw = [0.05, 0.1, 1, 0.25]

# number of sampling points
n_smp = 6000   # for velocity data
nh_smp = 5500  # for thickness data
n_col = 6000   # for collocation points
n_cbd = 300    # for boundary condition (calving front)
n_mbd = 1000   # for each interface between regions
# group all the number of points
n_pt = jnp.array([n_smp, nh_smp, n_col, n_cbd, n_mbd], dtype='int32')
# double the points for L-BFGS training
n_pt2 = jnp.array(n_pt*2, dtype='int32')


#%% data loading

# select the ice shelf for the training
shelfname = 'RnFlch'

# create the dataset filename
filename = 'data_xpinns_' + shelfname + '.mat'
filepath = project_root.joinpath('data').joinpath(filename)

# create the output file name
outputName = shelfname + f'_xpinns_aniso_seed={seed:.0f}'
# check whether sub-folder exists
outdir = rootdir.joinpath('Results_' + shelfname)
isExist = os.path.exists(outdir)
# create the sub-folder if not exist
if not isExist:
    os.mkdir(outdir)

# obtain the data for training
data_all, idxgall, posi_all, idxcrop_all = normalize_data(filepath)
scale = tree_map(lambda x: data_all[x][4][0:2], idxgall)


#%% initialization

# initialize the weights and biases of the network
trained_params = init_xpinns(keys[0], n_hl, n_unit, n_sub=len(idxgall), aniso=True)

# create the solution function [tuple(callable, callable)]
solNN = solu_create(scale)

# create the data function for Adam
dataf = data_sample_create(data_all, idxgall, n_pt)
keys_adam = random.split(keys[1], 5)
data = dataf(keys_adam[0])
# create the data function for L-BFGS
dataf_l = data_sample_create(data_all, idxgall, n_pt2)
key_lbfgs = keys[2]

# group the gov. eqn and bd cond.
eqn_all = (gov_eqn, front_eqn)
# calculate the loss function
NN_loss = loss_aniso_create(solNN, eqn_all, scale, idxgall, lw)
# calculate the initial loss and set it as the loss reference value
NN_loss.lref = NN_loss(trained_params, data)[0]


#%% networks training

# set the training iteration
epoch1 = 200000
epoch2 = 100000
# (above is the number of iterations required for high accuracy,
#  users are free to modify it based on your need)

# set the learning rate for Adam
lr = 1e-3
# define the scheduler function for weight of the regularization loss
wsp_schdul = lambda x: lax.max(10 ** (-x/epoch1 * 2), 0.0125)
# training the neural network
start_time = time.time()

"""training with Adam"""
trained_params, loss1 = adam_optimizer(
    keys_adam[0], NN_loss, trained_params, dataf, epoch1, lr=lr, aniso=True, schdul=wsp_schdul)

# sample the data for L-BFGS training
data_l = dataf_l(key_lbfgs)
"""training with L-BFGS"""
trained_params, loss2 = lbfgs_optimizer(NN_loss, trained_params, data_l, epoch2)

elapsed = time.time() - start_time
print('Training time: %.4f' % elapsed, file=sys.stderr)


#%% network saving

FileName = outputName + '.pkl'
FilePath = str(outdir.joinpath(FileName))
with open(FilePath, 'wb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it. However, users should have the same version of JAX
    # to load the data correctly.
    pickle.dump(trained_params, f, pickle.HIGHEST_PROTOCOL)


#%% prediction

# create the function for trained solution and equation residues
f_u = lambda x, idx: solNN[0](trained_params, x, idx)
# group all the function
func_all = (f_u, gov_eqn)
# calculate the solution and equation residue at given grids for visualization
results = predict(func_all, data_all, posi_all, idxcrop_all, idxgall, aniso=True)

# generate the last loss
loss_all = jnp.array(loss1 + loss2)
# save the loss info into results
results['loss'] = loss_all


#%% output saving

# save the output into .mat file
FileName = outputName + '.mat'
FilePath = str(outdir.joinpath(FileName))
savemat(FilePath, results)