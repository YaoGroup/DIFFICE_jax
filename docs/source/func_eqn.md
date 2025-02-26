# Equations and conditions

## Location: `diffice_jax/equation` 

The folder includes the codes that compute the residues of governing equations and boundary conditions 
involved in the PINN training. 
 
### `/eqn_iso.py`

involving functions to compute the residue of the normalized **isotropuc** Shallow-Shelf Approximation (SSA) 
equations and their associated dynamic boundary conditions at the calving front. Both the SSA equations and
the boundary conditions are given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Isotropic.md) .
 
### `/eqn_aniso_zz.py`

involving functions to compute the residue of the normalized **anisotropic** Shallow-Shelf Approximation (SSA) 
equations and their associated dynamic boundary conditions at the calving front. The suffix `_zz` indicate the
equation consider the anisotropic direction in the vertical direction.  The anisotorpic SSA equations and
the associated boundary conditions are given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/Anisotropic.md) .
