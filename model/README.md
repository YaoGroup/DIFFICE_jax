# MODEL for PINNs and XPINNs

This folder provides the core codes for generating neural network, creating
loss function and predicting output variables for the PINN training on assimilating 
remote-sensing data of ice shelves and infer their effective viscosity. Two versions
of codes are provided in this folder. The `pinns` folder involves the code is for 
the **regular PINN** training, and the `xpinns` folder is for the **extended-PINN
(XPINN)** training. 

The mathematical formation for the regular PINN training is provided in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/paper.md), 
and the description of the XPINNs setting is given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/model/XPINNs.md).
Both the `pinns` and `xpinns` folders contain four Python scripts, each specifying a key 
component for PINN training. Detailed instructions for correctly calling the functions in these 
scripts can be found in the [example codes](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) 
within the `examples` folder of this repository.


## Code description
### `initialization.py`

Involving essential functions to intialize weights and biases for all neural networks required 
for the problem. The code use Xavier initialization scheme, so that the weights between each 
two layers are generated following a truncated normal distribution with zero mean and the 
variance equal to $2/(n_{l-1}+n_{l})$, where $n_{l}$ and $n_{l+1}$ indicates the number of
units in the previous and next layers. The biases are initialized with all zero.


### `networks.py`

Involving essential functions to generate the neural network model for each physical variable 
involved in the problem. For regular PINN training, two networks are created. One network has 
three outputs, representing two velocity components and thickness. The other network has either 
one output for isotropic viscosity or two outputs for anisotropic viscosity components.  In
comparison, **XPINNs** generate two networks for each of the sub-region. Each network is a
fully-connected multiple-layer preceptrons (MLP) using `tanh` as the default activation function. 
When creating the networks, users need to specify whether the networks are created for isotropic 
and anistropic viscosity inference, as these two cases requires different number of outputs.


### `loss.py`

Involving essential functions to generate the total loss function for PINN training on assimilating
remote-sensing data of ice shelves and inferring their effective viscosity. The mathematical formation
of the loss function for inferring isotropic viscosity via regular PINNs is provided [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/paper.md). 
The description for inferring anisotorpic viscosity is given [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/examples/Anisotropic.md). 
The loss function for the XPINN training is given [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/model/XPINNs.md). 
We note that users should call different functions in the `loss.py` script to generate the loss functions
for inferring isotropic (i.e. `loss_iso_create()`) and anisotropic viscosity (i.e. `loss_aniso_create()`)


### `prediction.py`

Involving functions to predict the neural network output at the high-resolution grids for evaluation or 
visualization. The default setting of the function is to predict the data at the same resolution grid of
the remote-sensing velocity data (450m resolution). Users can modify the function to predict the network
output on other higher-resolution grids. In addition, the prediction function for **XPINNs**
will automatically stitch the network outputs from different sub-regions into a single large domain that 
covers all the sub-regions.
