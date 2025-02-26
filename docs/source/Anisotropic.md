# Anisotropic viscosity inversion

The conventional Shallow-shelf Approximation (SSA) equations is derived under the assumption that ice shelves have 
isotropic properties. However, evidence shows that ice shelves could exhibit anisotropic properties, leading to 
the viscosity inversion from the remote-sensing data and isotropic SSA equation with no viable solution.

## Governing equations

However, a fully-anisotropic viscosity matrix for ice shelves contains 16 components, making the viscosity
inversion problem ill-posed with no unique solution. Physically, considering the fact that the non-homogeneous
structure and crevasse of ice shelves are mainly present in the vertical direction, the viscosity associated 
with vertical deformation is likely to be most different from horizontal deformation. In that case, the 
anisotropic viscosity would only have two independent components, horizontal viscosity $\mu_h$ and vertical
viscosity $\mu_v$. 

Then, the modified shallow-shelf approximation in terms of the anisotropic viscosity components ($\mu_h$, $\mu_v$) can be derived as

$$\begin{eqnarray}
	 \frac{\partial} {\partial x}\left(2 {\mu_h} h \frac{\partial  u}{\partial x} + 2 {\mu_v} h \left[\frac{\partial  u}{\partial x}  + \frac{\partial  v}{\partial y} \right]  \right) 
	+ \frac{\partial} {\partial y} \left({\mu_h} h \left[\frac{\partial  u}{\partial y}  + \frac{\partial  v}{\partial x} \right]  \right)   = \rho g \left(1-{\frac{\rho}{\rho_w}}\right)h\frac{\partial h}{\partial x} \ , \qquad \\ 
	 \frac{\partial} {\partial y}\left(2 {\mu_h} h \frac{\partial  v}{\partial y} + 2 {\mu_v} h \left[\frac{\partial  u}{\partial x}  + \frac{\partial  v}{\partial y} \right]  \right)
  +\frac{\partial} {\partial x} \left({\mu_h} h \left[\frac{\partial  u}{\partial y}  + \frac{\partial  v}{\partial x} \right]  \right)  = \rho g \left(1-{\frac{\rho}{\rho_w}}\right)h\frac{\partial h}{\partial y}\ .   \qquad 
\end{eqnarray}$$
where $u$ and $v$ are the horizontal velocity, $h$ is the ice thickness and $\mu$ is the effective isotropic viscosity of the ice shelf. $\rho$ and $\rho_w$ are the density of the ice shelf and ocean water, respectively. $g$ is the gravity.

## Dynamic boundary conditions

The dynamic boundary conditions at the calving front with respect to the anisotropic viscosity components ($\mu_h$, $\mu_v$) becomes

$$\begin{array}{l}
\displaystyle 2\left({\mu_h}\frac{\partial u}{\partial x} + {\mu_v} \left[ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right] \right) n_x 
	+ \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_y 
 	= \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_x  \cr
\displaystyle  \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_x  
	+ 2 \left({\mu_h} \frac{\partial v}{\partial y} + {\mu_v} \left[\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right] \right) n_y 
 	= \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_y
\end{array} \quad \text{at} \  (x, y) \in {\partial \Omega_c} $$

where $\partial \Omega_c$ indicates the position of ice-shelf calving front, and $(n_x, n_y)$ is the unit normal 
vector towards outwards to the calving front.

It is important to note that the above governing equation and boundary conditions remain insufficient to guarantee 
the unique inversion of the two viscosity components ($\mu_h$, $\mu_v$). A **regularization condition** is, thus, required 
to ensure the uniqueness for inferring the anisotropic viscosity components ($\mu_h$, $\mu_v$). The regularization
condition we add is that we prefer the solutions of $\mu_h$ and $\mu_v$ are close to each other unless their 
closeness violates the SSA equations given the data of ice-shelf velocity and thickness. Practically, we can add an 
regularization term, which measures the mean squared error between the network prediction of $\mu_h$ and $\mu_v$ in 
the loss function of the PINN training, namely

$$\mathcal{L_{reg}} = \frac{\gamma_g}{N_c} {\sum^{N_c}_{i=1}} [\mu_h({\bf x_i}) - \mu_v({\bf x_i})]^2  $$

where ${\bf x_i}=(x_i, y_i)$ are the collocation points used to evaluate the value of $\mu_h$ and $\mu_v$ within the 
domain and $N_c$ is the total number of collocation points. Here, $\gamma_g$ is the hyper-parameter that represents 
the weight of the regularization loss in the loss function. Then, the total loss function for inferring anisotropic 
viscosity can be written as

$$ \begin{equation}
    \mathcal{L} = \mathcal{L_d} + \mathcal{L_e} (\gamma_e, \gamma_b) + \mathcal{L}_{reg}(\gamma_g)
\end{equation} $$

where $\mathcal{L_d}$ and $\mathcal{L_e}$ are the data loss and equation loss, respectively, which have the same expression
as for inferring [isotropic viscosity](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/source/Isotropic.md). $\gamma_e$ 
and $\gamma_b$ are the weighting pre-factors for the equation and boundary condition loss. Here, we note that the weight 
$\gamma_g$ for the regularization loss should be set to be much smaller than the weight $\gamma_e$ and $\gamma_b$. Otherwise, 
the contribution of the regularization loss overwhelms that of the equation loss in the loss function, causing PINNs to first
satisfy the regularization loss, rather than minimize the equation loss. This will gives the result of $\mu_h = \mu_v$, which 
is equavilent to inferring isotropic viscosity.
