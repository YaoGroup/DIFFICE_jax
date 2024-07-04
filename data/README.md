# Data

This folder contains several examples of remote-sensing velocity and thickness data for different 
ice shelves. The dataset for each ice shelf is directly cropped from the raw data files published online, 
maintaining the same resolution. No processing has been conducted on these data. The raw velocity data is downloaded
from NASA MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 [(NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1) 
with a 450m resolution. The thickness data is downloaded from NASA MEaSUREs BedMachine Antarctica, Version 2
[(NSIDC-0756)](https://nsidc.org/data/nsidc-0756/versions/2) with a 500m resolution.

Additionally, this folder provides the core codes that conduct the necessary preprocessing of the raw data,
including the normalization, reshaping and random sampling, which are all required to ensure a successful PINN
training. Two versions of codes are provided in this folder. The `pinns` folder involves the code is for the
**regular PINN** training,  and the `xpinns` folder is for the **extended-PINN (XPINN**) training.

 <br />
 
## Data Format

To be successfully loaded by the PINN code in this package, the raw data of each ice shelf needs to be organized 
and named in a particular way as shown in the example dataset in this folder. Users need to strictly follow the
same way if you want to create datasets for other ice shelves. The datasets for **PINN** and **XPINN** training 
shares certain similarity, but also contain differences that users need to pay attention to. 

First, the filename of the `.mat` dataset for both **PINNs** and **XPINNs** training should be made following
the form:

|| PINNs  | XPINNs|
| ------------- | ------------- | ------------ |
| Filename: | `data_pinns_#shelfname#.mat` |  `data_xpinns_#shelfname#.mat` |


Second, quantities with their names, physical meaning, and data types and shapes in the `.mat` dataset 
that are required for the **PINNs** or **XPINNs** training are listed in the table below:

|Variables | meaning | PINNs  | XPINNs|
| ------------- | ------------- | ------------ | ------------ |
| `xd` | x-position of **velocity** data | 2D matrix | cell[2D matrix]|
| `yd` | y-position of **velocity** data | 2D matrix | cell[2D matrix]|
| `ud` | velocity component along x-direction | 2D matrix | cell[2D matrix]|
| `vd` | velocity component along y-direction | 2D matrix | cell[2D matrix]|
| `xd_h` | x-position of **thickness** data | 2D matrix | cell[2D matrix]|
| `yd_h` | x-position of **thickness** data | 2D matrix | cell[2D matrix]|
| `hd` | thickness data | 2D matrix | cell[2D matrix]|
| `xct` | x-position of calving front | nx1 array | cell[nx1 array]|
| `yct` | y-position of calving front | nx1 array | cell[nx1 array]|
| `nnct` | unit normal vector of calving front | nx2 array | cell[nx2 array]|

The quantities listed above should be included in the datasets for both **PINN** and **XPINN** training. The
only difference is that, for XPINNs, each quantity should have a separate matrix for each sub-region, and
all of them should be saved in a `cell` type. Additionally, we note that all the velocity-related quantities (`xd`, `yd`,
`ud` and `vd`) should have the exact same shape, and the same applies to the thickness-related quantities (`xd_h`, `yd_h`
and `h_d`). The calving front-related quantities (`xct`, `yct` and `nnct`) should also have the same length. 

Apart from the above quantities, below are the additional quantities particularly required for the **XPINN** training

|Variables | meaning | XPINNs|
| ------------- | ------------- | ------------ |
| `x_md` | x-position of the interface between two sub-regions | cell[nx1 array]|
| `y_md` | x-position of the interface between two sub-regions | cell[nx1 array]|
| `Xe` | x-position of velocity data for the whole domain | 2D matrix|
| `Ye` | y-position of velocity data for the whole domain | 2D matrix|
| `Xe_h` | x-position of thickness data for the whole domain | 2D matrix|
| `Ye_h` | y-position of thickness data for the whole domain | 2D matrix|
| `idxcrop` | vertex position of sub-regional velocity data in the whole domain | cell[4x1 array]|
| `idxcrop_h` | vertex position of sub-regional thickness data in the whole domain | cell[4x1 array]|

`x_md` and `y_md` are the positions of the interface between each two consecutive sub-regions, which are required to 
evaluate the [continuity loss](https://github.com/YaoGroup/DIFFICE_jax/blob/main/model/XPINNs.md) for the **XPINN**
training. The other 6 quantities are the required information to merge the variable of all sub-regions into a 
large entity matrix, representing the whole domain.

 <br />

## Code description

### `DIFFICE_jax/data/preprocessing.py`

Involving essential functions to normalize the raw data loaded from the MATLAB format (`.mat`). The 
function in the script will automatically detect the characterisitc scale of each variable in the 
dataset, including the spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and use them to 
normalize those variables to be within the range of $[-1, 1]$. The script also re-organizes and reshapes
the data that is required by the code in the `model` folder to ensure the success of the PINN training.


### `DIFFICE_jax/data/sampling.py`

Involving essential functions to sample a batch of pre-processed data used for the PINN training. Users
can specify the number of sampling points for both velocity and thickness data, as well as the collocation 
points used to compute equation residue and boundary condition residue in each batch. Note that for
the sampling function in the `xpinns` folder, the number of sampling points specified by users is 
for each sub-region, not the entire domain.
