# Data processing

## Location: `diffice_jax/data` 

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

<br />


### `/pinns/preprocessing.py` / `/xpinns/preprocessing.py`

Involving essential functions to normalize the raw data loaded from the MATLAB format (`.mat`). The 
function in the script will automatically detect the characterisitc scale of each variable in the 
dataset, including the spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and use them to 
normalize those variables to be within the range of $[-1, 1]$. The script also re-organizes and reshapes
the data that is required by the code in the `model` folder to ensure the success of the PINN training.


<br />
 
### `/pinns/sampling.py` / `/xpinns/sampling.py`

Involving essential functions to sample a batch of pre-processed data used for the PINN training. Users
can specify the number of sampling points for both velocity and thickness data, as well as the collocation 
points used to compute equation residue and boundary condition residue in each batch. Note that for
the sampling function in the `xpinns` folder, the number of sampling points specified by users is 
for each sub-region, not the entire domain.
