# DIFFICE_jax
A user-friendly DIFFerentiable neural-network solver for data assimilation of ICE shelves written in JAX. 

## Introduction 
`DIFFICE_jax` is a Python package that solves the depth-integrated Stokes equation for **ice shelves**, and can be adopted for **ice sheets** by modifying the partial differential equations (PDE) in the neural network loss function. It uses PDEs to interpolate descretized remote-sensing data into meshless and differentible functions, and infer ice shelves' viscosity structure via back-propagation and automatic differentiation (AD). The algorithm is based on physics-informed neural networks [(PINNs)](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) and implemented in [JAX](https://jax.readthedocs.io/en/latest/index.html). The `DIFFICE_jax` package involves several advanced features in addition to vanilla PINNs algorithms, including collocation points resampling, non-dimensionalization of the data adnd equations, extended-PINNs [(XPINNs)](https://github.com/YaoGroup/DIFFICE_jax/blob/main/model/XPINNs.md) (see figure below), viscosity exponential scaling function, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The Github respository also provides [tutorial](https://github.com/YaoGroup/DIFFICE_jax/tree/main/tutorial) and real-data [examples](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) for users at different levels to have a good command of the package.


<img src="https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/figure/xpinns.png" alt="results" width="90%">


## Installation

The build of the code is tesed on Python version (3.9, 3.10 and 3.11) and JAX version (0.4.20, 0.4.23, 0.4.26)

You can install the package using pip as follows:
```python
python -m pip install DIFFICE_jax
```

## Documentation

The documentation for the algorithms and the mathematical formulation for the data assimilation of ice shelves 
are provided in the [`docs`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/docs)  folder. Documentations for the **tutorial** examples and **real-data** examples are  
given in the  [`tutorial`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/tutorial)  folder and [`examples`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) 
folders, respectively.

## Getting start with a Tutorial
We highly recommend the user who has no previous experience in either PINNs or inverse problems in Glaciology to get familar
with the software by reading the document and playing with the synthetic example prepared in the [`tutorial`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/tutorial) folder. The tutorial example
allow users to generate the synthetic data of velocity and thickness of an ice-shelf flow in a rectangular domain with any given 
viscosity profile. Users can then use the PINNs code prepared in the folder to infer the given viscosity from the synthetic code.
We provide a [Colab Notebook](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/tutorial/train_syndata.ipynb)
that allows users to compare the given viscosity with the PINN inferred viscosity to validate the accuracy of PINNs on inverse problem.


## Real-data Examples 
Besides the synthetic data in the `tutorial` folder, we provide the real velocity and thickness data for **four** different ice shelves surrounding the Antarctica
in the [`examples`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) folders. In the [paper](https://github.com/YaoGroup/DIFFICE_jax/tree/main/paper.md), we
summarized **six algorithm features** of the `DIFFICE_jax` package beyond the Vanilla PINNs code. Implementing different features, we provide four example codes in the 
[`examples`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) folders that can be used to analyze different ice-shelf datasets. 

For each example code, the corresponding implemented features and the ice-shelf dataset it can analyze are listed in the table below. All example codes are
well-documented in the [`examples`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) folder.

| Example codes  | Feature # | Ice shelf |
| ------------- | ------------- | ------------- |
| train_pinns_iso | (1), (2), (3), (4) | Amery, Larsen C, synthetic |
| train_pinns_aniso | (1), (2), (3), (4), (6)  | Amery, Larsen C|
| train_xpinns_iso  | (1), (2), (3), (4), (5)  | Ross, Ronne-Filchner|
| train_xpinns_aniso  | (1), (2), (3), (4), (5), (6)   |  Ross, Ronne-Filchner|

 <br />
 
## Google Colab
Apart from the Python scripts to run locally, we also provide **Colab Notebooks** for both the tutorial example and real
ice-shelf examples. They are provided in the [`tutorial`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/tutorial) and [`examples`](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) folders for a synthetic ice shelf and real ice shelves,
respectively. 

 <br />
 
## Diagram of Algorithm and Results.
<img src="https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/figure/PINN_setup.png" alt="results" width="90%">


## Contributors
This package is written by Yongji Wang and maintained by Yongji Wang (yongjiw@stanford.edu) and Ching-Yao Lai (cyaolai@stanford.edu). If you have questions about this code and documentation, or are interested in contributing the development of the `DIFFICE_jax` package, feel free to get in touch.  

## License
`DIFFICE_jax` is an open-source software. All code within the project is licensed under the MIT License. For more details, please refer to the [LICENSE](./LICENSE) file.

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
