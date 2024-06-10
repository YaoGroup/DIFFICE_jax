---
title: 'DIFFICE.JAX: Deep-learning-based differentiable solver for data assimilation and inverse modeling of ice shelves in JAX'
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
date: 14 May 2024
bibliography: paper.bib
---

# Summary

`DIFFICE.jax` is a user-friendly deep-learning based differentiable solver for ice-shelf dynamics written in JAX that can be used for data assimilation and viscosity inversion. It solves the depth-integrated Stokes equation for ice shelves and can be adopted for ice sheets. The prediction of ice dynamics relies on knowledge of its viscosity structure, which can not be directly measured at the Antarctic scale. Mis-representing viscsoity in ice-dynamics simulation can lead to imprecise forecasts of ice sheet's mass loss into the oceans and its consequent impact on global sea-level rise. With the continent-wide remote-sensing data available over the past decades, the viscosity of the ice shelves can be inferred by solving an inverse problem. We present `DIFFICE.jax`: Deep-learning-based DIFFerentiable solver for data assimilation and inverse modeling of ICE shelves in JAX, a Python package that convert descretized remote-sensing data into meshless and differentible functions, and infer the viscosity profile via PDE-constrained optimization and automatic differentiation (AD). The inversion algorithm is based on physics-informed neural networks (PINNs) [@Raissi2019] and implemented in JAX [@jax2018github]. The `DIFFICE.jax` package involves several advanced features in addition to vanilla PINNs algorithms, including collaction points resampling, non-dimensionalization of the data adnd equations, extended PINN, viscosity exponential scaling function, which are essential for accurate inversion. The package is designed to be user-friendly and accessible for beginners. The Github respository also provides tutorial examples for users at different levels to help master the method.


# Statement of need

The `DIFFICE.jax` package, along with several emerging deep-learning-based solver for ice-dynamics [@brinkerhoff2021constraining,@riel2021data,@riel2023variational,@wang2022discovering, @iwasaki2023one, @jouvet2022deep, @jouvet2023inversion, @jouvet2023ice,@bolibar2023universal,@he2023hybrid], advocates for the benefits of a differentiable solver leveraging GPUs for computational speedup. The `DIFFICE.jax` package is designed to make the inversion of ice-shelf viscosity via PINNs more accessible for beginners. Off-the-shelf PINNs code are not necessarily adequate for accurately inferring effective viscosity $\mu$ from the governing equations and real-world data. Additional settings for neural networks (NN), optimization methods, and pre-processing of both the SSA equations and observational data are all essential for the success of viscosity inversion via PINNs. The `DIFFICE.jax` package incorperates the optimal settings of PINNs in all these aspects. These settings are either universally applicable for training different ice shelves or can be determined automatically based on the data of given ice shelf. The package provides comprehensive details of the key algorithms involved, including comments and examples, enabling users to bypass substantial hyper-parameter tuning and conduct cutting-edge research in the field.

One of the unsolved questions in the cryosphere, or the broader field of geophysics, is how to uncover the hidden physical properties of various geophysical flows on Earth, such as glacial ice [@Millstein2022ice, ranganathan2024modified]. Ice shelves are the floating extensions of grounded ice sheet that play a critical role in slowing ice discharging into the ocean, mitigating the global sea level rise. A crtical physical properties required to predict ice dynamics is the effective ice viscosity. However, continent-wide in-situ measurement of the viscosity is challenging if not impossible. With high-resolution remote-sensing data available for both ice velocity $(u, v)$ and thickness $h$ across Antarctica, effective viscosity $\mu$ can be inferred via solving an inverse problem constrained by the governing equation of ice-shelf dynamics derived from the depth-integrated Stokes equations, i.e. the Shallow-Shelf Approximation (SSA) equations [@MacAyeal1989].

Due to the noises in the observational data, inverse problems are more difficult to solve than an forward problem. In cyrosphere, conventional method for inverse problems includes the control method, or called the adjoint method [[@MacAyeal1993, vieli2003application, @morlighem2010spatial, @goldberg2011data, @morlighem2013inversion, @furst2015assimilation], which requires regularization techniques to tackle the problem of error propagation from the data noise. The deep-learning based method, it has been demonstrated that the physics-informed neural networks (PINNs), can perform physics-based data de-noising while solving the inverse problem to infer ice rheology [@wang2022discovering, @riel2023variational, @iwasaki2023one] and sliding law [@riel2021data], without the need to add prior knowlegde of the viscosity or regularization terms in the loss function. The inherent properties of neural networks (NN) tends to denoise high-frequency error and noise involved in the data. Unlike the ajoint method, PINNs do not require the introduction of extra ajoint equations and associated adjoint boundary conditions [@MacAyeal1993, @morlighem2013inversion], which are usually challenging or time-consuming to derive for problems involving multiple coupled PDEs.


# Algorithm features

Critical features of `DIFFICE.jax` that go beyond off-the-shelf PINNs, and the necessity of these features to ensure the success and robustness of viscosity inference is explained below:

**(1) Data and equation normalization/non-dimensionalization:**
Proper training of NNs requires both input and output of the NN to be normalized, typically within the range of $[-1, 1]$. However, the values of observational data of ice velocity and thickness differ by several order of magnitude in their original units. Therefore, both their values (output) and spatial positions (input) need to be normalized before training. After normalizing the data, the new governing equations and associated boundary conditions, expressed in terms of the normalized quantities need to be re-derived. Each term in the new equation should have a magnitude of $O(1)$. The`DIFFICE.jax` package provides the algorithm that can automatically normalize (non-dimensionalize) the observational data and generate the associated normalized SSA equations for different ice shelves.

**(2) Optimal setting of equation weight:** 
The cost function of PINNs involves two terms: the data loss $\mathcal{L}_d$ and the equation loss $\mathcal{L}_e$. For viscosity inference, the data loss $\mathcal{L}_d$ quantifies the mismatch between the observed data and corresponding NN prediction $\mathrm{NN}_d$ via mean squared error, while the equation loss (or boundary condition loss) is defined as the mean squared error between the right and left-hand side of the equations (or boundary conditions). With the normalized data and equations, the data loss and equation loss are also normalized. The weighting pre-factors of the data and equation loss ($\gamma_b$ and $\gamma_e$ defined in the section below) are set to 1 and 0.1, respectively, optimized to minimize the training error and are verified to be universal for studying different ice shelves.

**(3) Design of NNs to enforce positive-definiteness:**
The effective viscosity $\mu$ must be positive everywhere. In addition, evidence shows that the spatial variation of $\mu$ within the ice shelf could cover several order of magnitude. We introduce the viscosity expression as 

$$\mu = \exp(\mathrm{NN_\mu}),$$ 

where $\mathrm{NN}_\mu$ is the output of the fully-connected NN created for $\mu$. This setting ensures the positive-definiteness of the inferred 
viscosity and enhance the training to capture both the local and global profile of viscosity with high accuracy over large spatial domain.

**(4) Residual-based re-sampling of collocation points during training:**
PINN training with observational data can often result in the cheating effects [@wang2022discovering, @charlie2024euler] due to errors and noise in the data. Here, cheating refers to the situation where the NN overfits the training data, leading to a small training loss but a large validation error elsewhere. 
A basic way to mitigate this issue is to randomly re-sample both data points and collocation points at regular intervals during training [@lu2021deepxde,@daw2022mitigating]. This can reduce the likelihood of overfitting. Collocation points refer to the points used to compute the equation residue. Additionally, a more effective approach to prevent overfitting and enhance training efficiency is to re-sample data and collocation points with higher concentration in areas where the spatial profile of the NN error or equation residue is larger. This residual-based resampling scheme is embedded in the `DIFFICE.jax` package as a default training setting.

**(5) Extended-PINNs (XPINNs) for large ice shelve:**
Large ice shelves, such as Ross, poses a multiscale challenge for capturing both local-scale and large-scale spatial variation of $u,v,h,\mu$. These local variations are difficult to capture with a single NN due to the spectral biases of NNs [@rahaman2019spectral,@xu2020frequency]. To address this challenge and ensure that PINNs capture those spatial variation precisely, the `DIFFICE.jax` package adopts the approach of extended PINNs (XPINNs) [@jagtap2020extended] for studying large ice shelves. This method divides the training domains into several sub-regions, with different NNs assigned to each. In this approach, each NN is trained to learn a specific sub-region of the large ice shelf, allowing it to capture local variations with high precision. We note that XPINNs require extra pre-processing of the observational data and additional constraints or penalty terms in the cost function to ensure successful training. Detailed requirements are documented in the `XPINNs` subfolder in the GitHub repository.

**(6) Inversion of anisotropic viscosity:**
Prior studies have shown that inversion of the isotropic viscosity can be over-constrained by remote-sensing data and the isotropic SSA equations [@wang2024deep], potentially leading to an ill-posed inverse problem with no viable solution. One way to address this over-constraint issue is to consider anisotropic viscosity. We find that the inversion with our modified SSA with anisotropic viscoisty can further reduce both the data and equation loss, indicating that the anisotropic equations are more consistent with the observations. The `DIFFICE.jax` package provides a comprehensive algorithm with well-posed settings for inferring anisotropic viscosity. The derivation of the anisotropic SSA equations, associated boundary conditions, and the additional loss terms in the cost functions to ensure the well-posedness of the inversion are described in detail in the `Anisotropic` subfolder in the GitHub repository. This is the first example of anisotropic-viscoisty inversion for ice shelves.

### Advantages
In addition to their ease of use, as mentioned above, `DIFFICE.jax` offer several advantages. First, the training of PINNs is effective even with irregularly sampled data, such as velocity data at a 450 m resolution [@Mouginot2019velo] and thickness data at a 500 m resolution [@Morlighem2020thick] that do not lie on the same grid. Second, while the outputs of classical numerical-based methods are discretized points that require enormous amounts of memory to store at high resolution, the outputs of PINNs are continuous functions parameterized by a fixed number of weights and biases, requiring relatively little memory even where higher resolutions is demanded [wang2024multi]. Third, instead of running several forward simulations iteratively until the loss function converges, the deep-learning based solver solves the forward and inverse problem simultaneaouly [Raissi2019]. The solver itself is *differentiable*; the gradient of the loss function with respect to the NN parameters are calculated via automatic differentiation (AD). This avoids the tedious efforts of writing the adjoint and is particularly advantageous when exploring new PDEs like the anisotropic equations. Benefits of automatic differentiation (AD) for glaciological inverse problems are also shown in other deep-learning based emulators [@jouvet2022deep,@jouvet2023inversion,@jouvet2023ice]. Fourth, PINN does not require an initial guess of $\mu$ to be included as prior in the cost function, as often used in the adjoint method for ice-hardness inversion [@furst2015assimilation, wang2022controls]. Finally, NN training tends to capture the main variations (low-frequency information) in the data, preventing disturbances from high-frequency errors or noise often present in the data. Therefore, no regularization techniques are required for PINN training.

# Examples

We provide a tutorial example that infers ice viscosity from synthetic data of ice velocity and thickness. Both datasets are created by solving the isotropic SSA equations and the mass conservation equation with a given viscosity profile numerically via COMSOL Multiphysics. The COMSOL file
is provided in the `Tutorial` subfolder. Users can freely generate new synthetic data by changing the given viscosity profile and test whether the PINN algorithm can accurately infer the correct viscosity profile from the synthetic data.

The ice-shelf viscosity inversion examples and the corresponding implemented features are listed in the table below. Well-documented codes can be found in their corresponding subfolders.

| Ice shelf  | Feature # |
| ------------- | ------------- |
| Synthetic ice shelf | (1), (2), (3), (4) |
| Amery ice shelf | (1), (2), (3), (4), (6)  |
| Larsen C ice shelf  | (1), (2), (3), (4), (6)  |
| Ross ice shelf  | (1), (2), (3), (4), (5), (6)   |
| Ronne-Filchner ice shelf  | (1), (2), (3), (4), (5), (6)  |

### Training data
The raw data used for the `DIFFICE.jax` package are downloaded from NASA MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 
[(NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1), for ice velocity and from NASA MEaSUREs BedMachine Antarctica, Version 2
[(NSIDC-0756)](https://nsidc.org/data/nsidc-0756/versions/2), for ice thickness. These raw datasets are not provided in the package. Instead, the datasets available in the package (all in the 'Data' subfolder) are cropped from the raw data under the same resolution and saved separately for each ice shelf. Additional information, such as the position of the ice-shelf calving front, required for the PINN training is also included in the dataset. The process of preparing the dataset from the raw data is documented in the `Data` subfolder, which help users create datasets for ice shelves not currently available in the package.

<!Question: existing library for convention method, (2) name of the libary. 
(3) what is the figure to add in the paper.!>


# Mathematical formulation

Ice shelf is a viscous gradient current. Due to the absence of shear stresses at both of its top and bottom surfaces, the motion of ice shelf 
is approximate to a two-dimensional flow, independent of the vertical direction. Assuming isotropic property, the ice-shelf dynamics is governed by the 2-dimensional Shallow-Shelf Approximation (SSA) equations [@MacAyeal1989], which read:

$$\begin{array}{l}
\displaystyle \frac{\partial} {\partial x}\left(4 \mu h \frac{\partial  u}{\partial x} + 2\mu h \frac{\partial  v}{\partial y}  \right) 
	+ \frac{\partial} {\partial y} \left( \mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x}  \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial x} \cr
\displaystyle  \frac{\partial} {\partial x} \left(\mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x} \right)
	+ \frac{\partial} {\partial y} \left( 2\mu h\frac{\partial u}{\partial x} + 4 \mu h\frac{\partial v}{\partial y} \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial y}
\end{array}$$

where $u$ and $v$ are the horizontal velocity, $h$ is the ice thickness and $\mu$ is the effective isotropic viscosity of the ice shelf. $\rho$ and $\rho_w$ are the density of the ice shelf and ocean water, respectively. $g$ is the gravity. The associated boundary conditions required for the equations is the *dynamic boundary condition* at the calving front of the ice shelf, which indicates the balance of the extensional stress of ice shelves with ocean hydrostatic pressure:

$$\begin{array}{l}
\displaystyle 2\mu \left(2\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) n_x 
	+ \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_y 
 	= \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_x  \cr
\displaystyle  \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_x  
	+ 2\mu \left(\frac{\partial u}{\partial x} + 2\frac{\partial v}{\partial y} \right) n_y 
 	= \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_y
\end{array} \quad \text{at} \  (x, y) \in {\partial \Omega_c} $$

where $\partial \Omega_c$ indicates the set of points at the calving front of the ice shelf, and $(n_x, n_y)$ is the unit normal vector towards outwards to the calving front. Both of the equations and boundary conditions needs to be involved in the PINN training for inferring isotropic ice viscosity. Thus, the cost function of the PINN training is expressed by

$$\mathcal{L} = \mathcal{L}_d + \mathcal{L}_e$$

where the data loss $\mathcal{L}_d$ reads

$$ \mathcal{L_d} = \frac{1}{N_d} \left( \sum_{i=1}^{N_d} [\hat{u_d} - u({\bf x_d})]^2 
	+ \sum_{i=1}^{N_d} [\hat{v_d} - v({\bf x_d})]^2 + \sum_{i=1}^{N_d} [\hat{v_d} - v({\bf x_d})]^2 \right) $$

where $\hat{u}_d$, $\hat{v}_d$ and $\hat{h}_d$ are the normalized data of ice-shelf velocity and thickness, respectively, at different normalized locations ${\bf \hat{x}_d}=(\hat{x}_d, \hat{y}_d)$. $N_d$ is the total number of points used for each iteration of the training. Then, the equation loss  $\mathcal{L}_e$ reads

$$ \mathcal{L_e} = \frac{\gamma_e}{N_e}\left(\sum_{i=1}^{N_e} [f_1({\bf \hat{x_e}})]^2 + 
	\sum_{i=1}^{N_e} [f_2({\bf \hat{x_e}})]^2 \right) 
 	+ \frac{\gamma_b}{N_b}\left(\sum_{i=1}^{N_b} [g_1({\bf \hat{x_b}})]^2 + \sum_{i=1}^{N_b} [g_2({\bf \hat{x_b}})]^2 \right)$$


where $f_1$ and $f_2$ represent the residues of the normalized isotropic SSA equations, and $g_1$ and $g_2$ are residues of the normalized dynamic boundary conditions. ${\bf \hat{x}_e}$ and ${\bf \hat{x}_b}$ are the normalized locations of collocations points to evaluate the residues of equations and boundary conditions, respectively. $N_e$ and $N_b$ are their total numbers. Here, the equation residue is the left-hand side minus of the right-hand side of the equation. $\gamma_e$ and $\gamma_b$ are the weighting pre-factors for the equation and boundary loss. With systematic test, we set $\gamma_b = 1$ around 10 times higher than $\gamma_e = 0.1$ for the success of training; PINNs priortize the solution that satisfies the boundary conditions, which guarantees the NN converges to a unique solution as long as the equation residue reduces.


# Figures

![Inferred viscosity for four different Antarctica Ice Shelves. \label{fig:example}](viscosity.png)


# Acknowledgements

We thank Charlie Cowen-Breen for the discussion on PINNs optimization, and
Ming-Ruey Chou and Robert Clapper for Python Programming and Github preparation.
We acknowledge the Office of the Dean for Research at Princeton University for partial 
funding support via the Dean for Research Fund for New Ideas in the Natural Sciences. 
C.-Y.L acknowledge the National Science Foundation for funding via Grant No. DMS-2245228.

# References
