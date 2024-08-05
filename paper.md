---
title: 'DIFFICE-jax: Differentiable neural-network solver for data assimilation of ice shelves in JAX'
tags:
  - Python
  - physics-informed deep learning
  - differentiable solver
  - JAX
  - geophysics
  - data assimilation 
  - ice-shelf dynamics
authors:
  - name: Yongji Wang
    orcid: 0000-0002-3987-9038
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Ching-Yao Lai
    orcid: 0000-0002-6552-7546
    affiliation: 1
affiliations:
 - name: Department of Geophysics, Stanford University, USA
   index: 1
 - name: Department of Mathematics, New York University, USA
   index: 2
date: 06 July 2024
bibliography: paper.bib
---

# Summary

The flow of Antarctic ice shelves is controlled by their viscosity structure, which cannot be directly measured at the continental scale. Misrepresenting viscosity in ice-dynamics simulations can lead to imprecise forecasts of ice sheet mass loss into the oceans and its consequential impact on global sea-level rise. With the continent-wide remote-sensing data available over the past decades, the viscosity of the ice shelves can be inferred by solving an inverse problem. We present DIFFICE_jax: a PINNs-based DIFFerentiable solver for data assimilation and inverse modeling of ICE shelves in JAX. This Python package converts discretized remote-sensing data into meshless and differentiable functions, and infers the viscosity profile by directly solving the Shallow-Shelf Approximation (SSA) equations for ice shelves. The inversion algorithm is implemented in JAX [@jax2018github]. The DIFFICE_jax package includes several advanced features beyond vanilla PINNs algorithms, such as collocation points resampling, non-dimensionalization of data and equations, extended-PINNs techniques, and function design of viscosity, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The GitHub repository also provides tutorial examples with Colab notebooks for users at different levels to reproduce the results and modify the code for their specific problems of interest.


# Statement of need

One of the unsolved questions in the cryosphere, or the broader field of geophysics, is how to uncover the hidden physical properties of various geophysical flows on Earth, such as ice-shelf flows [@Millstein2022ice; @ranganathan2024modified]. Ice shelves are the floating extensions of grounded ice sheet that play a critical role in slowing ice discharging into the ocean, mitigating the global sea level rise. A crtical physical properties required to predict ice dynamics is the effective ice viscosity. However, continent-wide in-situ measurement of the viscosity is challenging if not impossible. Instead, with high-resolution remote-sensing data available for both ice-shelf velocity $(u, v)$ and thickness $h$ across Antarctica, effective viscosity $\mu$ can be inferred via solving an inverse problem constrained by the Shallow-Shelf Approximation (SSA) equations [@MacAyeal1989].

Conventional method for solving inverse problems in glaciology includes the control method, or called the adjoint method [@MacAyeal1993; @vieli2003application; @morlighem2010spatial; @goldberg2011data; @morlighem2013inversion; @perego2014optimal; @furst2015assimilation; @barnes2021transferability], which is a PDE-constraint optimization method that requires deriving extra adjoint equations [@MacAyeal1993; @morlighem2013inversion]. Data used in inverse problems often contains noise. Thus, the adjoint method requires additional regularization techniques to prevent error propagation. In comparison, it has been demonstrated that the physics-informed neural networks (PINNs) [@raissi2019] can inherently de-noise the data while solving the inverse problem [@wang2022discovering; @riel2023variational; @iwasaki2023one] without the need of regularization terms in the cost function. Moreover, based on back-propagation and automatic differentiation, PINNs also do not require the derivation of ajoint equations, which is convenient for solving complex problems that involves multiple coupled PDEs. 

The `DIFFICE_jax` package is designed to make the inversion of ice-shelf viscosity via PINNs more accessible for beginners. Off-the-shelf PINNs code are not adequate for accurately inferring effective viscosity $\mu$ from the governing equations and real-world data. Additional settings for neural networks (NN), optimization methods, and pre-processing of both the SSA equations and observational data are all essential for the success of viscosity inversion via PINNs. The `DIFFICE_jax` package incorperates the optimal settings of PINNs in all these aspects. These settings are either universally applicable for training different ice shelves or can be determined automatically based on the data of given ice shelf. The package provides comprehensive details of the key algorithms involved, including comments and examples, enabling users to bypass substantial hyper-parameter tuning and conduct cutting-edge research in the field.

Another advantage of PINNs is its effectiveness with irregularly sampled data. The `DIFFICE_jax` package supports the direct use of thickness data from radar profiles available only at flight lines. In addition, the outputs of PINNs (velocity, thickness, and viscosity fields) are continuous functions parameterized by a fixed number of weights and biases, requiring less memory than the output of classical methods with discretized grid points when higher resolutions are demanded [@wang2024multi]. Finally, the `DIFFICE_jax` package, along with other deep-learning-based solver for ice-dynamics [@brinkerhoff2021constraining; @riel2021data; @riel2023variational; @wang2022discovering; @iwasaki2023one; @jouvet2022deep; @jouvet2023inversion; @jouvet2023ice; @bolibar2023universal; @he2023hybrid; @cheng2024unified], leverages GPUs for computational speedup. 


# Algorithm features

Key features of `DIFFICE_jax` that go beyond off-the-shelf PINNs, and the necessity of these features to ensure the success and robustness of viscosity inference are explained below:

**(1) Data and equation normalization/non-dimensionalization:**
Proper training of NNs requires both input and output of the NN to be normalized. However, the values of observational data of ice velocity and thickness differ by several order of magnitude in their original units. The `DIFFICE_jax` package provides the algorithm that can automatically normalize the observational data and derive normalized SSA equations for the training.

**(2) Optimal setting of equation weight:** 
The cost function of PINNs involves two terms: the data loss $\mathcal{L}_d$ and the equation loss $\mathcal{L}_e$. The weighting pre-factors of the data and equation loss are optimally set in the package to enure effective convergence of training for different ice shelves.

**(3) Design of NNs to enforce positive-definiteness:**
Considering that the effective viscosity $\mu$ is positive with large spatial variation within the ice shelf, we introduce the viscosity expression as 

$$\mu = \exp(\mathrm{NN_\mu}),$$ 

where $\mathrm{NN}_\mu$ is a fully-connected network for $\mu$. This setting ensures the positive-definiteness of the inferred 
viscosity and enhance the training to capture the large-varying viscosity profile with high accuracy.

**(4) Residual-based re-sampling of collocation points during training:**
Due to errors and noise, PINN training with observational data often cheats [@wang2022discovering; @charlie2024euler], where the networks overfit the training data. [@lu2021deepxde; @daw2022mitigating]. To prevent the issue, the `DIFFICE_jax` package uses residual-based re-sampling scheme during training, where more training data and collocation points are sampled in the area with large training residue.

**(5) Extended-PINNs (XPINNs) for large ice shelve:**
Regular PINN training with a single network cannot capture the rich spatial variation of large ice shelves, such as Ross. The `DIFFICE_jax` package adopts the approach of extended PINNs (XPINNs) [@jagtap2020extended] for studying large ice shelves. This method divides the training domains into several sub-regions, with different networks assigned to each. Detailed description of XPINNs are provided in the `model` folder of the GitHub repository.

**(6) Inversion of anisotropic viscosity:**
Prior studies have shown that the viscosity of Antarctica Ice Shelves could be anisotropic [@wang2024deep]. The `DIFFICE_jax` package involves the first algorithm designed to infer anisotropic viscosity. The governing equations, associated boundary conditions, and the cost function for inferring anisotropic viscosity are described in the `docs` folder. 

![**PINN setup**. (**a**) The structure and workflow of physics-informed neural networks (PINNs) for inferring ice viscosity $\mu$ from (**b**) the remote-sensing data of ice velocity $u,v$ [@Mouginot2019velo] and thickness $h$ [@Morlighem2020thick], and governing equations $(f_1,f_2,f_3,$ and $f_4=0)$. The loss function $\mathcal{L}$ contains two terms, the data loss $\mathcal{L}_d$ and the equation loss $\mathcal{L}_e$. (**c**) Prediction of trained neural network for velocity $(u, v)$ and thickness $h$, which shows high agreement with the remote-sensing data with relatively error around 1-3%. The inferred viscosity well satisfies the equation with small residue values, indicating the accuracy of the inferred viscosity.   \label{fig:PINN}](docs/PINN_setup.png)




# Acknowledgements

We thank Charlie Cowen-Breen for the discussion on PINNs optimization, and
Ming-Ruey Chou and Robert Clapper for Python Programming and Github preparation.
We acknowledge the Office of the Dean for Research at Princeton University for partial 
funding support via the Dean for Research Fund for New Ideas in the Natural Sciences. 
C.-Y.L acknowledge the National Science Foundation for funding via Grant No. DMS-2245228.

# References
