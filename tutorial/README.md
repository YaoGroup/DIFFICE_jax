[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/tutorial/pinn_syndata.ipynb)

# Tutorial example for viscosity inversion

This folder provide a tutorial example to infer the effective viscosity $\mu$ of ice shelves 
from synthentic data of ice velocity and thickness via physics-informed neural networks (PINNs).
Both of the simulation code for synthetic-data generation and the PINN code for viscosity
inversion are provided in this folder. All codes are well-documented for easy and good understanding.
In addition, we provided the ipynb file that allows the user to directly run the code online (on Colab).

## Problem setup
Considering the case of floating ice moving in a given domain, synthetic data refers to the
ice velocity and thickness data obtained by solving the Shallow-shelf Approximation 
equation and steady mass conservation equations numerically with a given viscosity, which reads

$$\begin{array}{l}
\displaystyle \frac{\partial} {\partial x}\left(4 \mu h \frac{\partial  u}{\partial x} + 2\mu h \frac{\partial  v}{\partial y}  \right) 
	+ \frac{\partial} {\partial y} \left( \mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x}  \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial x} \qquad \text{(SSA)} \cr
\displaystyle  \frac{\partial} {\partial x} \left(\mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x} \right)
	+ \frac{\partial} {\partial y} \left( 2\mu h\frac{\partial u}{\partial x} + 4 \mu h\frac{\partial v}{\partial y} \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial y} \qquad \text{(SSA)} \cr
\end{array}$$

$$ \qquad \frac{\partial u} {\partial x} + \frac{\partial v} {\partial y} = 0 \qquad \qquad \text{(Steady mass conserv.)} $$

where $(u, v)$, $h$ and $\mu$ are the velocity vector, thickness and viscosity of the floating ice, 
respectively. For the tutorial example in this folder, we consider the floating ice moving in a 
confined rectangular domain. The domain size and the associated boundary conditions of the ice flow are listed
in the figure below. Combining the equations and boundary conditions, the 

![Boundary conditions for generating synthetic data](COMSOL/IceShelf2D_bd.png){width=20%}




