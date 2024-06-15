import sys
import os
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
import time
from scipy.io import savemat
from pathlib import Path
import pickle

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.sampling import data_sample_create
from equation.ssa_eqn_iso import vectgrad, gov_eqn, front_eqn
from model.initialization import init_MLP
from model.networks import solu_create
from model.loss import loss_create
from model.prediction import predict
from optimizer.optimizer import adam_optimizer, lbfgs_optimizer
from tutorial.load_syndata import iceshelf_data


# find the root directory
rootdir = Path(__file__).parent

#%% setting the hyper-parameters

# select the random seed
seed = np.random.choice(3000, 1)[0]
key = random.PRNGKey(seed)
np.random.seed(seed)

# create the subkeys
keys = random.split(key, 4)

# select the size of neural network
n_hl = 5
n_unit = 30
lw = [0.05, 0.1]

# number of sampling points
n_smp = 4000
n_col = 4000
n_cbd = 600
n_pt = jnp.array([n_smp, n_col, n_cbd], dtype='int32')
n_pt2 = n_pt * 2   # double the points for L-BFGS

# create the dataset filename
DataFile = 'SynData_exp1.mat'
OutputName = f'SynData_SGD_idx{seed: .0f}'
# check whether sub-folder exists
outdir = rootdir.joinpath('Results')
isExist = os.path.exists(outdir)
# create the sub-folder if not exist
if isExist is False:
    os.mkdir(outdir)

# obtain the data for training
data_all = iceshelf_data(DataFile, step=1)
scale = data_all[4][0:2]

# initialize the weights and biases of the network
trained_params = init_MLP(keys[0], n_hl, n_unit)

# create the solution function
pred_u = solu_create()

# create the data function for Adam
dataf = data_sample_create(data_all, n_pt)
keys_adam = random.split(keys[1], 5)
data = dataf(keys_adam[0])
# create the data function for L-BFGS
dataf_l = data_sample_create(data_all, n_pt2)
key_lbfgs = keys[2]

# group the gov. eqn and bd cond.
eqn_all = (gov_eqn, front_eqn)
# calculate the loss function
NN_loss = loss_create(pred_u, eqn_all, scale, lw)
# update the reference value of the loss
NN_loss.lref = NN_loss(trained_params, data)[0]


#%% training the networks

# set the training iteration
epoch1 = 10000
epoch2 = 10000

"""Training using Adam"""

# set the learning rate for Adam
lr = 1e-3
# training the neural network
start_time = time.time()
# training with Adam
trained_params, loss1 = adam_optimizer(keys_adam[0], NN_loss, trained_params, dataf, epoch1, lr=lr)

# data_l = dataf_l(key_lbfgs)
# # training with L-bfgs
# trained_params, loss2 = lbfgs_optimizer(NN_loss, trained_params, data_l, epoch2)

elapsed = time.time() - start_time
print('Training time: %.4f' % elapsed)

#%% save the trained networks parameters

FileName = OutputName + '.pkl'
FilePath = str(outdir.joinpath(FileName))
with open(FilePath, 'wb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle.dump(trained_params, f, pickle.HIGHEST_PROTOCOL)


#%% prediction

# create the function for trained solution and equation residues
f_u = lambda x: pred_u(trained_params, x)
f_gu = lambda x: vectgrad(f_u, x)[0][:, 0:6]
f_eqn = lambda x: gov_eqn(f_u, x, scale)
# group all the function
func_all = (f_u, f_gu, f_eqn)
# calculate the solution and equation residue at given grids for visualization
results = predict(func_all, data_all)

# generate the last loss
loss_all = jnp.array(loss1)
# save the loss info into results
results['loss'] = loss_all


#%% Saving the results

# convert all the results into numpy format before saving
output = tree_map(lambda x: np.array(x), results)

# save the output into .mat file
FileName = OutputName + '.mat'
FilePath = str(outdir.joinpath(FileName))
savemat(FilePath, output)
