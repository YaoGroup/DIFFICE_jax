# Data source

The remote-sensing data of both ice-shelf velocity and thickness that can be used for the `DIFFICE_jax` package 
are available online The raw velocity data can be downloaded from NASA MEaSUREs Phase-Based Antarctica Ice Velocity Map, 
Version 1 [(NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1) with a 450m resolution. The thickness data can be
downloaded from NASA MEaSUREs BedMachine Antarctica, Version 2 [(NSIDC-0756)](https://nsidc.org/data/nsidc-0756/versions/2)
with a 500m resolution. These raw datasets are not provided in the package. Instead, the datasets provided in the `examples`
folder are cropped from these raw datasets under the same resolution with no extra processing, and saved separately for each
ice shelf. Additional information that is required for the PINN training,  such as the position of the ice-shelf calving front,
is also included in the dataset. The full requirement of the dataset that can ensure the success of PINN training is provided below. 
Users should strictly follow these requirements in order to create datasets for ice shelves that are not currently available 
in the package.


 <br />
 
## Data Format

To be successfully loaded by the PINN code in this package, the input data of each ice shelf needs to be organized 
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

Apart from the above quantities, below are the additional quantities particularly required for the **XPINNs** training

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
evaluate the [continuity loss](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/XPINNs.md) for the **XPINNs**
training. The other 6 quantities are the required information to merge the variable of all sub-regions into a 
large entity matrix, representing the whole domain.

 <br />

