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
