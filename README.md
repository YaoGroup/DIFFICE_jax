# DIFFICE_jax
Neural network-based differentiable solver for data assimilation and inverse modeling of ice shelves in JAX.
`DIFFICE_jax` is a user-friendly deep-learning based differentiable solver for ice-shelf dynamics written in JAX that can be used for data assimilation and viscosity inversion. It solves the depth-integrated Stokes equation for ice shelves and can be adopted for ice sheets. The prediction of ice dynamics relies on knowledge of its viscosity structure, which can not be directly measured at the Antarctic scale. Mis-representing viscsoity in ice-dynamics simulation can lead to imprecise forecasts of ice sheet's mass loss into the oceans and its consequent impact on global sea-level rise. With the continent-wide remote-sensing data available over the past decades, the viscosity of the ice shelves can be inferred by solving an inverse problem. We present `DIFFICE_jax`: Neural-network-based DIFFerentiable solver for data assimilation and inverse modeling of ICE shelves in JAX, a Python package that convert descretized remote-sensing data into meshless and differentible functions, and infer the viscosity profile via PDE-constrained optimization and automatic differentiation (AD). The inversion algorithm is based on physics-informed neural networks (PINNs) [@Raissi2019] and implemented in JAX [@jax2018github]. The `DIFFICE_jax` package involves several advanced features in addition to vanilla PINNs algorithms, including collaction points resampling, non-dimensionalization of the data adnd equations, extended PINN, viscosity exponential scaling function, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The Github respository also provides tutorial examples for users at different levels to help master the method.

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
