---
title: 'Deep-learning-based differentiable solver for data assimilation and physics inversion of ice shelves'
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
contribution to rising sea levels is ice loss from Antarctica through the 
collapse of ice shelves. However, the fundamental mechanical properties, such 
as viscosity and flow law of ice shelves, have been highly debated for 
over half a century. Mis-representing these properties can lead to imprecise 
forecasts of ice mass loss into the oceans and its consequent impact on global 
sea-level rise. With the continent-wide remote-sensing data available over the
past decades, the viscosity and flow law of the entire ice shelves can be inferred
from these data as an inverse problem. We present `pinnIceShelf` as a python package
that can conduct data assimilation that convert descretized remote-sensing data to
meshless and differentible functions, and to further infer the viscosity profile
from that. The inversion algorithm is based on the Physics-informed Neural 
Networks (PINNs) and written in JAX. The `pinnIceShelf` respository provided 
a tutorial example for users to better understand and implement the method.


# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Ice shelf is a viscous gradient current that has slender-body shape. Due to the absence 
of shear stresses at both of its top and bottom surfaces, the motion of ice shelf 
is approximate to a two-dimensional flow, independent of the vertical direction. 
Assuming isotropic property, the ice-shelf dynamics is governed by the 2-D
Shallow-Shelf Approximation (SSA) equations, which read:

$$\begin{array}{l}
\displaystyle \frac{\partial} {\partial x}\left(4 \mu h \frac{\partial  u}{\partial x} + 2\mu h \frac{\partial  v}{\partial y}  \right) 
	+ \frac{\partial} {\partial y} \left( \mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x}  \right)   = \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial x} \cr
\displaystyle  \frac{\partial} {\partial x} \left(\mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial  v}{\partial x} \right) + \frac{\partial} {\partial y} \left( 2\mu h\frac{\partial u}{\partial x} + 4 \mu h\frac{\partial v}{\partial y} \right)  = \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial y}
\end{array}$$

where $u$ and $v$ are the horizontal velocity, $h$ is the ice thickness and $\mu$ is the effective isotropic viscosity of the ice shelf. $\rho$ and $\rho_w$ are the density of the ice shelf and ocean water, respectively. $g$ is the gravity. The associated boundary conditions required for the equations is the *dynamic boundary condition* at the calving front of the ice shelf, which indicates the balance of the extensional stress of ice shelves with ocean hydrostatic pressure. It gives,

$$\begin{array}{l}
\displaystyle 2\mu \left(2\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) n_x + \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_y = \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_x  \cr
\displaystyle  \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_x  + 2\mu \left(\frac{\partial u}{\partial x} + 2\frac{\partial v}{\partial y} \right) n_y  =  \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_y
\end{array} \quad \text{at} \  (x, y) \in {\partial \Omega_c} $$

where $\partial \Omega_c$ indicates the set of points at the calving front of the ice shelf, and $(n_x, n_y)$ is the local unit normal vector towards outwards to the calving front.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

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
