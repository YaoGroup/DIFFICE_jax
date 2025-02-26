# PINNs and XPINNs models

## Location: `diffice_jax/model`

This folder provides the core codes for generating neural network, creating
loss function and predicting output variables for the PINN training on assimilating 
remote-sensing data of ice shelves and infer their effective viscosity. Two versions
of codes are provided in this folder. The `pinns` folder involves the code is for 
the **regular PINN** training, and the `xpinns` folder is for the **extended-PINN
(XPINN)** training. 

<br /> 


### `diffice_jax/model/pinns/initialization.py` / `diffice_jax/model/xpinns/initialization.py`

Involving essential functions to intialize weights and biases for all neural networks required 
for the problem. The code use Xavier initialization scheme, so that the weights between each 
two layers are generated following a truncated normal distribution with zero mean and the 
variance equal to $2/(n_{l-1}+n_{l})$, where $n_{l}$ and $n_{l+1}$ indicates the number of
units in the previous and next layers. The biases are initialized with all zero.


<br /> 


### `diffice_jax/model/pinns/networks.py` / `diffice_jax/model/xpinns/networks.py`

Involving essential functions to generate the neural network model for each physical variable 
involved in the problem. For regular PINN training, two networks are created. One network has 
three outputs, representing two velocity components and thickness. The other network has either 
one output for isotropic viscosity or two outputs for anisotropic viscosity components.  In
comparison, **XPINNs** generate two networks for each of the sub-region. Each network is a
fully-connected multiple-layer preceptrons (MLP) using `tanh` as the default activation function. 
When creating the neural networks, users need to specify whether the neural networks are created for isotropic 
and anistropic viscosity, as these two cases requires different number of outputs.


 <br /> 


### `diffice_jax/model/pinns/loss.py` / `diffice_jax/model/xpinns/loss.py`

Involving essential functions to generate the total loss function for PINN training on assimilating
remote-sensing data of ice shelves and inferring their effective viscosity. The mathematical formation
of the loss function for inferring isotropic viscosity via regular PINNs is provided [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Isotropic.md). 
The description for inferring anisotorpic viscosity is given [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Anisotropic.md). 
The loss function for the XPINN training is given [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/XPINNs.md). 
We note that users should call different functions in the `loss.py` script to generate the loss functions
for inferring isotropic (i.e. `loss_iso_create()`) and anisotropic viscosity (i.e. `loss_aniso_create()`)

 <br /> 

### `diffice_jax/model/pinns/prediction.py` / `diffice_jax/model/xpinns/prediction.py`

Involving functions to predict the neural network output at the high-resolution grids for evaluation or 
visualization. The default setting of the function is to predict the data at the same resolution grid of
the remote-sensing velocity data (450m resolution). Although users can modify the function to predict the network
output on other higher-resolution grids. Furthermore, the prediction function for **XPINNs**
will automatically stitch the network outputs from different sub-regions into a single large domain that 
covers all the sub-regions.
