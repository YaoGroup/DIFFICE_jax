# MODEL for PINNs and XPINNs

This folder provides the core codes for generating neural network, creating
loss function and predicting output variables for the PINN training on assimilating 
remote-sensing data of ice shelves and infer their effective viscosity. Two versions
of codes are provided in this folder. The `pinns` folder involves the code is for 
the **regular PINN** training, and the `xpinns` folder is for the **extended-PINN
(XPINN)** training. 

The mathematical formation for the regular PINN training is provided in [this link], 
and the description of the XPINNs setting is given in [this link]. Either `pinns` 
or `xpinns` folder involves four python scripts that specify four key components
for the PINN training, respectively

## Code description
### `initialization.py`

Involving python functions to intialize weights and biases for all neural networks required 
for the problem. The code use Xavier initialization scheme, so that the weights between each 
two layers are generated following a truncated normal distribution with zero mean and the 
variance equal to $2/(n_{l-1}+n_{l})$, where $n_{l}$ and $n_{l+1}$ indicates the number of
units in the previous and next layers. The biases are initialized with all zero.


### `networks.py`

Involving python functions to generate the neural network model for each physical variable
involved in the problem. For regular PINN training, two networks are created. One network 
has three outputs, representing two velocity components and thickness. The other network has 
either one output for isotropic viscosity, or two outputs for two anisotropic viscosity 
components. In comparison, **XPINNs** generate two networks for each of the sub-region. Each
network are fully-connected multiple-layer preceptrons (MLP) using tanh activation function
as default. 


### `loss.py`

Generate the total loss function for PINN training on assimilating remote-sensing data of ice
shelves and inferring their effective viscosity. 
