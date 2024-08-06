# DIFFICE_jax
A user-friendly DIFFerentiable neural-network solver for data assimilation of ICE shelves written in JAX. 

## Introduction 
`DIFFICE_jax` is a Python package that solves the depth-integrated Stokes equation for **ice shelves**, and can be adopted for **ice sheets** by modifying the partial differential equations (PDE) in the neural network loss function. It uses PDEs to interpolate descretized remote-sensing data into meshless and differentible functions, and infer ice shelves' viscosity structure via back-propagation and automatic differentiation (AD). The algorithm is based on physics-informed neural networks [(PINNs)](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) and implemented in [JAX](https://jax.readthedocs.io/en/latest/index.html). The `DIFFICE_jax` package involves several advanced features in addition to vanilla PINNs algorithms, including collocation points resampling, non-dimensionalization of the data adnd equations, extended-PINNs [(XPINNs)](https://github.com/YaoGroup/DIFFICE_jax/blob/main/model/XPINNs.md) (see figure below), viscosity exponential scaling function, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The Github respository also provides [tutorial](https://github.com/YaoGroup/DIFFICE_jax/tree/main/tutorial) and real-data [examples](https://github.com/YaoGroup/DIFFICE_jax/tree/main/examples) for users at different levels to have a good command of the package.

<p align="center">
    <img src="https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/figure/xpinns.png" alt="results" width="90%">
</p>

## Installation

Instructions are for installation into a virtual Python Environment. Please ensure that Python 3.x has been installed in your 
local machine or the remote compute machine, such as HPC cluster or Google Cloud Platform (GCP). We recommend the Python of 
version later than 3.9.0. 

1. Create a virtual environment named `DIFFICE_jax`
```python
python -m venv DIFFICE_jax
```

2. Activate the Virtual Environment (for MacOS/linux)
```python
source DIFFICE_jax/bin/activate
```

3. Install JAX. The package only works for JAX version 0.4.23 or later.
```python
# Install JAX on CPU (not recommended, too slow)
pip install jax[cpu]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install JAX on GPU (recommended if GPUs are available)
pip install jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. Install other Python Dependencies required for the package
```python
# required for Adam optimizer
pip install optax

# required for L-BFGS optimizer
pip install tfp-nightly

# for output ploting
pip install matplotlib
```

5. Clone the `DIFFICE_jax` package locally from GitHub
```python
git clone https://github.com/YaoGroup/DIFFICE_jax.git
```

6. Run the example codes
```python
# tutorial example using synthetic data
python3 DIFFICE_jax/tutorial/train_syndata.py

# example using real data of ice shelves
python3 DIFFICE_jax/examples/train_pinns_iso.py
```

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
