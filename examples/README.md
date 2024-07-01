# Examples

This folder provides four example codes that use `DIFFICE_jax` package to assimilate remote-sensing velocity
and thickness data and infer effective ice viscosity under either **isotropic** or **anisotropic** assumption
via regular PINNs or extended-PINNs (XPINNs).

## `train_pinns_aniso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under anisotropic
assumption via regular PINNs. The code are computationally-efficient and accurate enough to study ice shelves
of size close or smaller than Amery or Larce C Ice Shelves. An companion Colab Notebook of this script is 
provided in the `colab` subfolder or can be opened directly by clicking here
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/tutorial/train_syndata.ipynb)

