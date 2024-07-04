# DIFFICE_jax
A user-friendly neural-network-based DIFFerentiable solver for data assimilation and inverse modeling of ICE shelves written in JAX. 

## Introduction 
`DIFFICE_jax` is a Python package that solves the depth-integrated Stokes equation for `ice shelves`, and can be adopted for `ice sheets` by modifying the partial differential equations (PDE) in the neural network loss function. It uses PDEs to interpolate descretized remote-sensing data into meshless and differentible functions, and infer ice shelves' viscosity structure via `PDE-constrained optimization` and `automatic differentiation` (AD). The algorithm is based on physics-informed neural networks (`PINNs`) [@Raissi2019] and implemented in JAX [@jax2018github]. The `DIFFICE_jax` package involves several advanced features in addition to vanilla PINNs algorithms, including collocation points resampling, non-dimensionalization of the data adnd equations, extended PINN, viscosity exponential scaling function, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The Github respository also provides tutorial examples for users at different levels to help master the method.

Please direct questions about this code and documentation to Yongji Wang (yongjiw@stanford.edu) and Ching-Yao Lai (cyaolai@stanford.edu).

## Installation

Instructions are for installation into a virtual Python Environment. Please ensure that Python 3.x has been installed in your 
local machine or the remote compute machine, such as HPC cluster or Google Cloud Platform (GCP). We recommend the Python of 
version later than 3.9.0. 

Create a virtual environment named `diffice_jax`

```python
python -m venv diffice_jax
```

Activate the Virtual Environment (for MacOS/linux)

```python
source diffice_jax/bin/activate
```

Install JAX. The package only works for JAX version 0.4.23 or later.

```python
# Install JAX on CPU (not recommended, too slow)
pip install jax[cpu]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install JAX on GPU (recommended if GPUs are available)
pip install jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Install other Python Dependencies required for the package


Clone the `DIFFICE_jax` package locally from GitHub using

```python
git clone https://github.com/YaoGroup/DIFFICE_jax.git
```





## Citation
BibTex:
```
@article{wang2022discovering,
  title={Discovering the rheology of Antarctic Ice Shelves via physics-informed deep learning},
  author={Wang, Yongji and Lai, Ching-Yao and Cowen-Breen, Charlie},
  year={2022},
  doi = {https://doi.org/10.21203/rs.3.rs-2135795/v1},
}
```
