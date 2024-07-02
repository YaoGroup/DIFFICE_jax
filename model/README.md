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
### `Initialization.py`
