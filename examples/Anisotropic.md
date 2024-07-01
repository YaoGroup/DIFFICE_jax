# Anisotropic viscosity inversion via PINNs

The conventional Shallow-shelf Approximation (SSA) equations is derived under the assumption that ice shelves have 
isotropic properties. However, evidence shows that ice shelves could exhibit anisotropic properties, leading to 
the viscosity inversion from the remote-sensing data and isotropic SSA equation with no viable solution.

However, a fully-anisotropic viscosity matrix for ice shelves contains 16 components, making the viscosity
inversion problem ill-posed with no unique solution. Physically, considering the fact that the non-homogeneous
structure and crevasse of ice shelves are mainly present in the vertical direction, the viscosity associated 
with vertical deformation is likely to be most different from horizontal deformation. In that case, the 
anisotropic viscosity would only have two independent components, horizontal viscosity $\mu_h$ and vertical
viscosity $\mu_v$. 

Then, the modified shallow-shelf approximation in terms of the anisotropic viscosity tensor \eqref{eq:muhv} can be derived to be

$$\begin{eqnarray}
	 \frac{\partial} {\partial x}\left(2 {\mu_h} h \frac{\partial  u}{\partial x} + 2 {\mu_v} h \left[\frac{\partial  u}{\partial x}  + \frac{\partial  v}{\partial y} \right]  \right) 
	+ \frac{\partial} {\partial y} \left({\mu_h} h \left[\frac{\partial  u}{\partial y}  + \frac{\partial  v}{\partial x} \right]  \right)   = \rho g \left(1-{\frac{\rho}{\rho_w}}\right)h\frac{\partial h}{\partial x} \ , \qquad \\ 
	 \frac{\partial} {\partial y}\left(2 {\mu_h} h \frac{\partial  v}{\partial y} + 2 {\mu_v} h \left[\frac{\partial  u}{\partial x}  + \frac{\partial  v}{\partial y} \right]  \right)
  +\frac{\partial} {\partial x} \left({\mu_h} h \left[\frac{\partial  u}{\partial y}  + \frac{\partial  v}{\partial x} \right]  \right)  = \rho g \left(1-{\frac{\rho}{\rho_w}}\right)h\frac{\partial h}{\partial y}\ .   \qquad 
\end{eqnarray}$$

Similarly, the dynamic boundary conditions at the calving front with respect to the anisotropic viscosity components ($\mu_h$, $\mu_v$) becomes

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
the unique inversion of the two viscosity components ($\mu_h$, $\mu_v$). A regularization condition is, thus, required 
to ensure the uniqueness for inferring the anisotropic viscosity components ($\mu_h$, $\mu_v$). That regularization
condition we add is that we prefer the solutions of $\mu_h$ and $\mu_v$ are close to each other unless their 
closeness violates the SSA equations given the data of ice-shelf velocity and thickness. Practically, we can add an 
regularization term, which measures the mean squared error between the network prediction of $\mu_h$ and $\mu_v$ in 
the loss function of the PINN training, namely

$$ \begin{equation}
    \mathcal{L}_{reg} = \frac{1}{N_c} \sum_{i=1}^{N_c} [\mu_h({\bf x_c}) - \mu_v({\bf x_c})]^2
\end{equation} $$

where ${\bf x_c}=(x_c, y_c)$ are the collocation points used to evaluate the value of $\mu_h$ and $\mu_v$ within the 
domain and $N_c$ is the total number of collocation points.
