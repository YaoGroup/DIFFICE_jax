# DIFFICE_jax
A user-friendly neural-network-based DIFFerentiable solver for data assimilation and inverse modeling of ICE shelves written in JAX. 

`DIFFICE_jax` is a Python package that solves the depth-integrated Stokes equation for `ice shelves`, and can be adopted for `ice sheets` by modifying the partial differential equations (PDE) in the neural network loss function. It uses PDEs to interpolate descretized remote-sensing data into meshless and differentible functions, and infer ice shelves' viscosity structure via `PDE-constrained optimization` and `automatic differentiation` (AD). The algorithm is based on physics-informed neural networks (`PINNs`) [@Raissi2019] and implemented in JAX [@jax2018github]. The `DIFFICE_jax` package involves several advanced features in addition to vanilla PINNs algorithms, including collaction points resampling, non-dimensionalization of the data adnd equations, extended PINN, viscosity exponential scaling function, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The Github respository also provides tutorial examples for users at different levels to help master the method.

Please direct questions about this code and documentation to Yongji Wang (yongjiw@stanford.edu) and Ching-Yao Lai (cyaolai@stanford.edu).

# Citation
BibTex:
```
@article{wang2022discovering,
  title={Discovering the rheology of Antarctic Ice Shelves via physics-informed deep learning},
  author={Wang, Yongji and Lai, Ching-Yao and Cowen-Breen, Charlie},
  year={2022},
  doi = {https://doi.org/10.21203/rs.3.rs-2135795/v1},
}
```
