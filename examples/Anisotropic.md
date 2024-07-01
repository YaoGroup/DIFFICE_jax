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

Then, the modified shallow-shelf approximation in terms of the anisotropic viscosity tensor \eqref{eq:muhv} can be derived (see Appendix \ref{sec:aniso}) to be

$$\begin{eqnarray}
	 \frac{\partial} {\partial x}\left(2 {\mu_h} h \frac{\partial  u}{\partial x} + 2 {\mu_v} h \left[\frac{\partial  u}{\partial x}  + \frac{\partial  v}{\partial y} \right]  \right) 
	+ \frac{\partial} {\partial y} \left({\mu_h} h \left[\frac{\partial  u}{\partial y}  + \frac{\partial  v}{\partial x} \right]  \right)   = \rho g \left(1-{\frac{\rho}{\rho_w}}\right)h\frac{\partial h}{\partial x} \ , \qquad \\ 
	 \frac{\partial} {\partial y}\left(2 {\mu_h} h \frac{\partial  v}{\partial y} + 2 {\mu_v} h \left[\frac{\partial  u}{\partial x}  + \frac{\partial  v}{\partial y} \right]  \right)
  +\frac{\partial} {\partial x} \left({\mu_h} h \left[\frac{\partial  u}{\partial y}  + \frac{\partial  v}{\partial x} \right]  \right)  = \rho g \left(1-{\frac{\rho}{\rho_w}}\right)h\frac{\partial h}{\partial y}\ .   \qquad 
\end{eqnarray}$$

Similarly, the corresponding dynamic boundary conditions at the calving front with respect to the anisotropic viscosity components ($\mu_h$, $\mu_v$) becomes (see Appendix \ref{sec:aniso})

$$\begin{eqnarray}
	2\left({\mu_h}\frac{\partial u}{\partial x} + \colb{\mu_v} \left[ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right] \right) n_x + \colr{\mu_h} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_y = \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_x   \\ 
	{\mu_h} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) n_x  + 2 \left(\colr{\mu_h} \frac{\partial v}{\partial y} + \colb{\mu_v} \left[\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right] \right) n_y  =  \frac{1}{2}\rho g h\left(1 - \frac{\rho}{\rho_w} \right)  n_y 
\end{eqnarray}$$
