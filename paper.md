---
title: 'Deep-learning-based differentiable solver for data assimilation and inverse modeling of ice shelves'
tags:
  - Python
  - physics-informed deep learning
  - geophysics
  - ice-shelf dynamics
authors:
  - name: Yongji Wang
    orcid: 0000-0002-3987-9038
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Ching-Yao Lai
    orcid: 0000-0002-6552-7546
    affiliation: 2
affiliations:
 - name: Department of Mathematics, New York University, USA
   index: 1
 - name: Department of Geophysics, Stanford University, USA
   index: 2
date: 14 May 2024
bibliography: paper.bib
---

# Summary

Sea-level rise is one of the most serious implications of climate change (IPCC) 
and will impact the lives of hundreds of millions of people. A dominant 
contributor to rising sea levels is ice loss from Antarctica due to the 
collapse of ice shelves. However, the fundamental mechanical properties, such 
as viscosity and flow law of ice shelves, have been highly debated for 
over half a century. Mis-representing these properties can lead to imprecise 
forecasts of ice mass loss into the oceans and its consequent impact on global 
sea-level rise. With the continent-wide remote-sensing data available over the
past decades, the viscosity and flow law of the entire ice shelves can be inferred
from these data as an inverse problem. We present `pinnIceShelf` as a Python package
that can conduct data assimilation to convert descretized remote-sensing data into
meshless and differentible functions, and further infer the viscosity profile
from them. The inversion algorithm is based on physics-informed Neural 
Networks (PINNs) [@Raissi2019] and implemented in JAX [@jax2018github]. The 
`pinnIceShelf` package involves advanced features in addition to regular PINNs
algorithms, which are essential for solving inversion problem correctly. The package
is designed to be user-friendly and accessible for beginners. The Github respository
also provides tutorial examples for users at different levels to help master the method.


# Statement of need

One of the main research focus of cryoshphere, or broader field of geophyscs
is to uncover the hidden physical properties of various geophysical flows 
on Earth, such as Antarctica Ice Shelves. Ice shelves are the floating extensions 
of grounded ice sheet, and play a critical role in slowing ice discharging 
into the ocean, thereby mitigating the global sea level rise. A crtical physical
properties of Ice Shelves that governs their initial discharging from the ground
to their final breakup into the ocean is the effective ice viscosity, which is also 
essential for understanding the flow law of ice shelves. However, the large length 
scales and long time scales of ice-shelf flow make the in-situ measurement
of the viscosity significantly challenging anc costly. Alternatively, with
high-resolution data available for both ice velocity $(u, v)$ and thickness $h$ 
across Antarctica, effective viscosity can be inferred as an inverse
problem by solving the Shallow-Shelf Approximation (SSA) equations, which governs
the ice-shelf dynamics.

Due to the potential high-level error or noise in the observational data, inverse 
problems are more difficult to solve than an forward problem. In cyrosphere, 
conventional method for inverse problems includes control method [@MacAyeal1993], 
or called adjoint method, which is, however, quite challenging to implement or 
generalize for various problems, especially for beginners with less knowledge in 
applied mathematics or numerical methods. Additionally, a newly-developed 
deep-learning based method, named physics-informed neural networks (PINNs), are shown 
to be efficient in solving inverse problem. Unlike the ajoint method, PINNs do not 
require the introduction of extra ajoint equations and associated adjoint boundary
conditions, which are usually difficult to derive for complicated problems. 
Moreover, the inherent properties of neural networks tends to denoise high-frequency 
error and noise involved in the data. Thus, there is no need to employ regularization 
techqniues, such as adding regularization terms in the cost function, when solving 
inverse problems.

However, to successfully infer effective viscosity from the SSA equations and 
remote-sensing data, regular PINNs codes or algorithms available online might be 
inadequate. Additional settings for neural networks, optimization methods, and 
pre-processing of both the SSA equations and observational data are all essential
for the success of viscosity inversion via PINNs. The `pinnIceShelf` package 
incorperates the optimal settings of PINNs in all these aspects. These settings are
either universally applicable for training different ice shelves or can be determined 
automatically based on the data of given ice shelf. Overall, The `pinnIceShelf`
package is designed to make the inversion of ice-shelf viscosity more accessible for
beginners with less knowledge or experience on PINNs. On the other end, the package
provides comprehensive details of the key algorithms involved, including comments and
examples, that can help users conduct cutting-edge research in the field.

# Features and advantages

Critical features of `pinnIceShelf` that go beyond regular PINNs and are essential for
viscosity inference includes: (1) data and equation normalization; (2) optimal setting of 
equation weight; (3) design of networks with positive-definiteness; (4) Residual-based
re-sampling of points during training; (5) Extended-PINNs (XPINNs) for studying large
ice shelves; (6) Inversion of anisotropic viscosity. The necessity of these features to 
ensure the success of viscosity inference is explained below.

First, proper training of neural networks requires both input and output of the network
to be normalized, typically within the range of $[-1, 1]$. However, the values of 
observational data of ice velocity and thickness differ by several order of magnitude
in their original units. Therefore, both their values (output) and spatial positions 
(input) need to be normalized before training. After normalizing the data, the new SSA
equations and associated boundary conditions, expressed in terms of the normalized 
quantities need to be re-derived, and each term should have a magnitude of $O(1)$. The
`pinnIceShelf` package provides the algorithm that can automatically normalize the data
and equations for all different ice shelves.

Second, the cost function of PINNs involves two terms: the data loss $\mathcal{L}_d$ 
and the equation loss $\mathcal{L}_e$. For viscosity inference, the data loss 
$\mathcal{L}_d$ quantifies the mismatch between the observed data and neural network 
approximation $\mathrm{NN}_d$ via mean squared error, while the equation loss (or 
boundary condition loss) is defined as the mean squared error between the right and 
left-hand side of the equations (or boundary conditions). With the normalized data and
equations, the data loss and equation loss are also normalized. The pre-factor of data 
loss is often set to 1. The value for the prefactor of the equation loss 
and the boundary condition loss are optimized to minimize the training error and are 
verified to be universal for studying different ice shelves.

Third, effective viscosity $\mu$ is a physical quantity that must be positive 
everywhere. In addition, evidence shows that the spatial variation of $\mu$ within the 
ice shelf could cover several order of magnitude. Considering these two properties of
viscosity $\mu$ to be captured by PINNs, we introduce the viscosity expression as 
$\mu = \exp(\mathrm{NN_\mu})$ where $\mathrm{NN}_\mu$ is the output of the fully-connected
network created for $\mu$. This setting ensures the positive-definiteness of the inferred 
viscosity and enhance the training to capture both the local and global profile of 
viscosity with high accuracy over large spatial domain.

Fourth, PINN training with observed data can often result in cheating due to errors 
and noise in the data. Here, cheating refers to the situation where the network overfits
the training data, leading to a small training loss but a large validation error elsewhere. 
A basic way to mitigate this issue is to randomly re-sample both data points and 
collocation points at regular intervals during training. This can reduce the likelihood 
of overfitting. Collocation points refer to the points used to compute the equation residue. 
Additionally, a more effective approach to prevent overfitting and enhance training 
efficiency is to re-sample data and collocation points with higher concentration in areas 
where the spatial profile of the network error or equation residue is larger. This 
residual-based resampling scheme is embedded in the `pinnIceShelf` package as a default
training setting.

Fifth, large ice shelves, such as Ross, contain local structural provinces where physical
quantities often have dense local variations. These local variations are difficult to capture
with a single neural network due to the spectral biases of networks [@rahaman2019spectral].
To address this challenge and ensure that PINNs capture those spatial variation precisely, 
the pinnIceShelf package adopts the approach of extended physics-informed neural networks 
(XPINNs) [@jagtap2020extended] for studying large ice shelves. This method divides the 
training domains into several sub-regions, with different networks assigned to each. In 
this approach, each network is trained to learn a specific sub-region of the large ice shelf,
allowing it to capture local variations with high precision. We note that XPINNs require 
extra pre-processing of the observational data and additional constraints or penalty terms
in the cost function to ensure successful training. Detailed requirements are documented 
in the `XPINNs` subfolder in the GitHub repository.

Sixth, prior studies have shown that inferring the isotropic viscosity of ice shelves 
can be over-constrained by remote-sensing data and the isotropic SSA equations, leading 
to an ill-posed inverse problem with no viable solution. One reason for this over-constraint
issue is that ice-shelf viscosity is actually anisotropic. The `pinnIceShelf` package
provides a comprehensive algorithm with well-posed settings for inferring anisotropic 
viscosity. The derivation of the anisotropic SSA equations, associated boundary conditions,
and the additional loss terms in the cost functions to ensure the well-posedness of the 
inversion are described in detail in the `Anisotropic` subfolder in the GitHub repository.

The combination of the above six features ensures that PINNs are reliable tools for 
inferring ice viscosity. In addition to their ease of use, as mentioned above, PINNs
offer several \textit{advantages} over the classical control method for solving inverse 
problems.

First, the training of PINNs is effective even with irregularly sampled data, such as 
velocity data at a 450 m resolution [@Mouginot2019velo] and thickness data at a 500 m 
resolution [@Morlighem2020thick] that do not lie on the same grid. Second, 
while the outputs of classical numerical-based methods are discretized points that require
enormous amounts of memory to store at high resolution, the outputs of PINNs are continuous
functions parameterized by a fixed number of weights and biases, requiring relatively little
memory even where higher resolutions is demanded. Third, computing derivatives of discretized
points from either observed data or numerical-based methods is challenging and often induce 
extra error. In contrast, once the network, as a continuous function representation, is 
trained to approximate the target function, we can compute the \textit{exact} derivative of 
the network output via automatic differentiation. This is extremely useful, for example, 
when computing the effective strain rate, which involves the spatial derivative of the 
ice velocity. Fourth, neural network training tends to capture the main variations 
(low-frequency information) in the data, preventing disturbances from high-frequency errors
or noise often present in the data. Therefore, no regularization techniques are required
for PINN training.

# Accessibility

To make the approach of inferring ice-shelf viscosity via PINNs more convincible, 
understandable and accessible to users in general, we provide a tutorial example that
infers the ice viscosity from the synthetic data of ice velocity and thickness. Both
datasets are created by solving the isotropic SSA equations and mass conservation 
equation with a given viscosity profile numerically via COMSOL Multiphysics. The COMSOL 
file is provided in the `Tutorial` subfolder. Users can freely generate the new synthetic
data by changing the given viscosity profile and test whether the PINN algorithm can 
infer the correct viscosity profile as given from the synthetic data. We note that the
tutorial example only contains the first four features of `pinnIceShelf`. For the last 
two advanced features: (5) extended-PINNs approach and (6) inversion of anisotropic 
viscosity, well-documented examples for selected ice shelves are provided in their 
corresponding subfolders to help users employ or further generalize the methods.

<!Question: existing library for convention method, (2) name of the libary. 
(3) what is the figure to add in the paper.!>



# Mathematics

Ice shelf is a viscous gradient current that has slender-body shape. Due to the absence 
of shear stresses at both of its top and bottom surfaces, the motion of ice shelf 
is approximate to a two-dimensional flow, independent of the vertical direction. 
Assuming isotropic property, the ice-shelf dynamics is governed by the 2-D
Shallow-Shelf Approximation (SSA) equations, which read:

$$\begin{array}{l}
\displaystyle \frac{\partial} {\partial x}\left(4 \mu h \frac{\partial  u}{\partial x} + 2\mu h \frac{\partial  v}{\partial y}  \right) 
	+ \frac{\partial} {\partial y} \left( \mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x}  \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial x} \cr
\displaystyle  \frac{\partial} {\partial x} \left(\mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial  v}{\partial x} \right)
	+ \frac{\partial} {\partial y} \left( 2\mu h\frac{\partial u}{\partial x} + 4 \mu h\frac{\partial v}{\partial y} \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial y}
\end{array}$$

where $u$ and $v$ are the horizontal velocity, $h$ is the ice thickness and $\mu$ is 
the effective isotropic viscosity of the ice shelf. $\rho$ and $\rho_w$ are the density 
of the ice shelf and ocean water, respectively. $g$ is the gravity. The associated 
boundary conditions required for the equations is the *dynamic boundary condition* at
the calving front of the ice shelf, which indicates the balance of the extensional 
stress of ice shelves with ocean hydrostatic pressure. It gives,

$$\begin{array}{l}
\displaystyle 2\mu \left(2\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) n_x + \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_y = \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_x  \cr
\displaystyle  \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_x  + 2\mu \left(\frac{\partial u}{\partial x} + 2\frac{\partial v}{\partial y} \right) n_y  =  \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_y
\end{array} \quad \text{at} \  (x, y) \in {\partial \Omega_c} $$

where $\partial \Omega_c$ indicates the set of points at the calving front of the ice shelf, and $(n_x, n_y)$ is the local unit normal vector towards outwards to the calving front.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

=
For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
