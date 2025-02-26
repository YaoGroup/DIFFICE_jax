# Isotropic viscosity inversion

Ice shelf is a viscous gravity current. Due to the absence of shear stresses at both of its top and bottom surfaces, the motion of ice shelf 
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

where $\partial \Omega_c$ indicates the set of points at the calving front of the ice shelf, and $(n_x, n_y)$ is the unit normal vector towards outwards to the calving front. Both of the equations and boundary conditions needs to be involved in the PINN training for inferring isotropic ice viscosity. Thus, the loss function of the PINN training is expressed by

$$\mathcal{L} = \mathcal{L}_d + \mathcal{L}_e$$

where the data loss $\mathcal{L}_d$ reads

$$ \mathcal{L_d} = \frac{1}{N_d} \left( \sum_{i=1}^{N_d} [\hat{u_d} - u({\bf \hat{x_d}})]^2 
	+ \sum_{i=1}^{N_d} [\hat{v_d} - v({\bf \hat{x_d}})]^2 + \sum_{i=1}^{N_d} [\hat{v_d} - v({\bf \hat{x_d}})]^2 \right) $$

where $\hat{u}_d$, $\hat{v}_d$ and $\hat{h}_d$ are the normalized data of ice-shelf velocity and thickness, respectively, at different normalized locations ${\bf \hat{x}_d}=(\hat{x}_d, \hat{y}_d)$. $N_d$ is the total number of points used for each iteration of the training. Then, the equation loss  $\mathcal{L}_e$ reads

$$ \mathcal{L_e} = \frac{\gamma_e}{N_e}\left(\sum_{i=1}^{N_e} [f_1({\bf \hat{x_e}})]^2 + 
	\sum_{i=1}^{N_e} [f_2({\bf \hat{x_e}})]^2 \right) 
 	+ \frac{\gamma_b}{N_b}\left(\sum_{i=1}^{N_b} [g_1({\bf \hat{x_b}})]^2 + \sum_{i=1}^{N_b} [g_2({\bf \hat{x_b}})]^2 \right)$$


where $f_1$ and $f_2$ represent the residues of the normalized isotropic SSA equations, and $g_1$ and $g_2$ are residues of the normalized dynamic boundary conditions. ${\bf \hat{x}_e}$ and ${\bf \hat{x}_b}$ are the normalized locations of collocations points to evaluate the residues of equations and boundary conditions, respectively. $N_e$ and $N_b$ are their total numbers. Here, the equation residue is the left-hand side minus of the right-hand side of the equation. $\gamma_e$ and $\gamma_b$ are the weighting pre-factors for the equation and boundary loss. With systematic test, we set $\gamma_b = 1$ around 10 times higher than $\gamma_e = 0.1$ for the success of training; PINNs priortize the solution that satisfies the boundary conditions, which guarantees the NN converges to a unique solution as long as the equation residue reduces.
