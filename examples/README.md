# Examples

This folder provides four example codes that use `DIFFICE_jax` package to assimilate remote-sensing velocity
and thickness data and infer effective ice viscosity under either **isotropic** or **anisotropic** assumption
via regular PINNs or extended-PINNs (XPINNs).

## `train_pinns_aniso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **anisotropic**
assumption via **regular PINNs**. The code are computationally-efficient and accurate enough to study ice shelves
of size close or smaller than Amery or Larce C Ice Shelves. An companion Colab Notebook of this script is 
provided in the `colab` subfolder or can be opened directly by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_pinns_aniso.ipynb)


## `train_pinns_iso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **isotropic**
assumption via **regular PINNs**. The code are computationally-efficient and accurate enough to study ice shelves
of size close or smaller than Amery or Larce C Ice Shelves. An companion Colab Notebook of this script is 
provided in the `colab` subfolder or can be opened directly by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_pinns_iso.ipynb)


## `train_xpinns_aniso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **anisotropic**
assumption via **extended-PINNs (XPINNs)**. This code are required to several largest ice shelves around the
Antarctica, such as Ross and Ronne-Filchner Ice Shelves, which involve many local structural regions with dense 
spatial variation that are difficult to be captured by one single neural network due to the spectral biases.
An companion Colab Notebook of this script is provided in the `colab` subfolder or can be opened directly by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_xpinns_aniso.ipynb)

