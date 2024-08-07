# Code description


## Folder: `diffice_jax/data` 
This folder provides the codes that conduct the pre-preprocessing of the input data,
including the normalization, flattening and random sampling, and the post-processing of the neural
network output, including re-normalization and reshaping. Two versions of codes are provided in this 
folder. The `pinns` subfolder involves the code is for the **regular PINNs** training,  and the `xpinns` 
subfolder is for the **extended-PINNs (XPINNs**) training.

The mathematical formation for the **regular PINNs** training is provided in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Isotropic.md), 
and the description of the **XPINNs** setting is given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/XPINNs.md).
Both the `pinns` and `xpinns` folders contain four Python scripts, each specifying a key 
component for PINN training. Detailed instructions for correctly calling the functions in these 
scripts can be found in the [example codes](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) 
within the `examples` folder of this repository.

 
### `diffice_jax/data/pinns/preprocessing.py` / `diffice_jax/data/xpinns/preprocessing.py`

Involving essential functions to normalize the raw data loaded from the MATLAB format (`.mat`). The 
function in the script will automatically detect the characterisitc scale of each variable in the 
dataset, including the spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and use them to 
normalize those variables to be within the range of $[-1, 1]$. The script also re-organizes and reshapes
the data that is required by the code in the `model` folder to ensure the success of the PINN training.

 
### `diffice_jax/data/pinns/sampling.py` / `diffice_jax/data/xpinns/sampling.py`

Involving essential functions to sample a batch of pre-processed data used for the PINN training. Users
can specify the number of sampling points for both velocity and thickness data, as well as the collocation 
points used to compute equation residue and boundary condition residue in each batch. Note that for
the sampling function in the `xpinns` folder, the number of sampling points specified by users is 
for each sub-region, not the entire domain.


 <br />
 <br />
 

## Folder: `diffice_jax/equation` 

The folder includes the codes that compute the residues of governing equations and boundary conditions 
involved in the PINN training. 
 
### `diffice_jax/equation/eqn_iso.py`

involving functions to compute the residue of the normalized **isotropuc** Shallow-Shelf Approximation (SSA) 
equations and their associated dynamic boundary conditions at the calving front. Both the SSA equations and
the boundary conditions are given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Isotropic.md) .
 
### `diffice_jax/equation/eqn_aniso_zz.py`

involving functions to compute the residue of the normalized **anisotropic** Shallow-Shelf Approximation (SSA) 
equations and their associated dynamic boundary conditions at the calving front. The suffix `_zz` indicate the
equation consider the anisotropic direction in the vertical direction.  The anisotorpic SSA equations and
the associated boundary conditions are given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Anisotropic.md) .

 <br />
 <br />

 

## Folder: `diffice_jax/model`

This folder provides the core codes for generating neural network, creating
loss function and predicting output variables for the PINN training on assimilating 
remote-sensing data of ice shelves and infer their effective viscosity. Two versions
of codes are provided in this folder. The `pinns` folder involves the code is for 
the **regular PINN** training, and the `xpinns` folder is for the **extended-PINN
(XPINN)** training. 


### `diffice_jax/model/pinns/initialization.py` / `diffice_jax/model/xpinns/initialization.py`

Involving essential functions to intialize weights and biases for all neural networks required 
for the problem. The code use Xavier initialization scheme, so that the weights between each 
two layers are generated following a truncated normal distribution with zero mean and the 
variance equal to $2/(n_{l-1}+n_{l})$, where $n_{l}$ and $n_{l+1}$ indicates the number of
units in the previous and next layers. The biases are initialized with all zero.


### `diffice_jax/model/pinns/networks.py` / `diffice_jax/model/xpinns/networks.py`

Involving essential functions to generate the neural network model for each physical variable 
involved in the problem. For regular PINN training, two networks are created. One network has 
three outputs, representing two velocity components and thickness. The other network has either 
one output for isotropic viscosity or two outputs for anisotropic viscosity components.  In
comparison, **XPINNs** generate two networks for each of the sub-region. Each network is a
fully-connected multiple-layer preceptrons (MLP) using `tanh` as the default activation function. 
When creating the neural networks, users need to specify whether the neural networks are created for isotropic 
and anistropic viscosity, as these two cases requires different number of outputs.

 
### `diffice_jax/model/pinns/loss.py` / `diffice_jax/model/xpinns/loss.py`

Involving essential functions to generate the total loss function for PINN training on assimilating
remote-sensing data of ice shelves and inferring their effective viscosity. The mathematical formation
of the loss function for inferring isotropic viscosity via regular PINNs is provided [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Isotropic.md). 
The description for inferring anisotorpic viscosity is given [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Anisotropic.md). 
The loss function for the XPINN training is given [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/XPINNs.md). 
We note that users should call different functions in the `loss.py` script to generate the loss functions
for inferring isotropic (i.e. `loss_iso_create()`) and anisotropic viscosity (i.e. `loss_aniso_create()`)

 
### `diffice_jax/model/pinns/prediction.py` / `diffice_jax/model/xpinns/prediction.py`

Involving functions to predict the neural network output at the high-resolution grids for evaluation or 
visualization. The default setting of the function is to predict the data at the same resolution grid of
the remote-sensing velocity data (450m resolution). Although users can modify the function to predict the network
output on other higher-resolution grids. Furthermore, the prediction function for **XPINNs**
will automatically stitch the network outputs from different sub-regions into a single large domain that 
covers all the sub-regions.

 <br />
 <br />

 
## Folder: `diffice_jax/optimizer`

### `diffice_jax/optimizer/optimization.py`

Providing provides two optimization methods:

[**Adam**](https://arxiv.org/pdf/1412.6980): a first-order gradient-based optimization method 
based on adaptive estimates of lower-order moments.

[**L-BFGS**](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf): Limited-memory BFGS method, a second-order quasi-Newton optimization method for
solving unconstrained nonlinear optimization problems, using a limited amount of computer memory.

A **stochastic training scheme** is applied to two optimizers, where the code randomizes both data samples and collocation points at regular interval during training 
to minimize the cheating effect.
