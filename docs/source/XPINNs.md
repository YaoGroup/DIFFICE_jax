# Why we need XPINNs

Large ice shelves surrounding the Antarctica, such as Ross and Ronne Ice Shelves, contain many local structural provinces,
including suture zones, margin shear zones, and fast-flowing zones, causing local variations of their physical properties, such as viscosity. However, neural networks training exhibits [spectral bias](https://proceedings.mlr.press/v97/rahaman19a/rahaman19a.pdf), 
for which the neural networks tend to learn low-frequency information of the target function, while omitting the high-frequency one.
Considering a large ice shelf as a singje domain, those local variations thus become high-frequency information that
is hard to capture by a single neural network. 

To resolve this issuse, the **extended-PINNs (XPINNs)** separate any large domain into several sub-regions (see the figure)
and arrange different networks for them. In that case, each network only need to capture the physical variables within 
a small local region, largely increase the capacity for predicting the multiscale variation of different physical variables.

Compared to regular PINN training, the key difference of XPINN training is to impose the continuity conditions at the
interface (blue line in the figure) between consecutive sub-regions that ensure the neural network predictions to be
continuous and smooth across the sub-regions. The continuity condition is also essential for the uniqueness of the 
inferred quantities from PINNs.

<p align="center">
    <img src="../figure/xpinns.png" alt="results" width="90%">
</p>

# Loss function for XPINNs

To ensure that the neural networks prediction of two consecutive sub-regions remain continuous at their interface, 
Several continuity loss terms needs to be added in the cost function. The first continuity loss term is given as

$$ \begin{eqnarray}
    \mathcal{L_{c0}} &=& \sum_{j=1}^{N_s} \frac{1}{N_{\Omega_j}} \left[\sum_{i=1}^{N_{\Omega_j}} [u_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - u_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 + \sum_{i=1}^{N_{\Omega_j}} [v_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - v_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 + \right. \\
    & & \left.\sum_{i=1}^{N_{\Omega_j}} [h_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - h_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 + \sum_{i=1}^{N_{\Omega_j}} [\mu_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \mu_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 \right]
\end{eqnarray} $$

where ${\bf \hat{x_{\Omega_j}}}=(\hat{x_{\Omega_j}},\ \hat{y_{\Omega_j}})$ are the normalized locations for the 
collocation points on the $j$-th interface $\Omega_j$ between two adjacent networks and $N_{\Omega_j}$ is their total
number. $q_{j}^{(+)}$ and $q_{j}^{(-)}$ represent the neural network predictions in the two sub-regions that intersect
at the $j$-th interface $\Omega_j$, where $q$ stands for $u$, $v$, $h$ or $\mu$. $N_s$ is the total number of the 
interfaces between two adjacent networks. 

In addition, the SSA equations involves higher order derivatives of different physical variables. Thus, we require 
the higher derivatives of those variables (up to the highest order that appears in the equations) also to be continuous
across the interface. Recalling that the SSA equation involves the second order derivative of the ice velocity $u$ and
$v$, and the first derivative of the ice thickness $h$ and the viscosity $\mu$, the additional continuity loss terms
required include

$$ \begin{eqnarray}
    \mathcal{L_{c1}} = \sum_{j=1}^{N_s} \frac{1}{N_{\Omega_j}} & &\left[
    \sum_{i=1}^{N_{\Omega_j}} ||\nabla u_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \nabla u_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})||^2 
    + \sum_{i=1}^{N_{\Omega_j}} ||\nabla v_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \nabla v_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})||^2 \right. \\ 
    & & \left.\sum_{i=1}^{N_{\Omega_j}} || \nabla h_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \nabla_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})||^2 + 
    \sum_{i=1}^{N_{\Omega_j}} ||\nabla \mu_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \nabla\mu_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})||^2 
    \right],  
\end{eqnarray} $$

which matches the first derivative of all the physical variables involved in the SSA equations at the interfaces. Here, 
$\nabla = (\partial_x, \partial_y)$ is the gradient operator and $||{\bf v}||$ indicates the norm of the vector ${\bf v}$.
To match the second order derivative of ice velocity $u$ and $v$ at the interfaces, we have

$$ \begin{eqnarray}
    \mathcal{L_{c2}} = \sum_{j=1}^{N_s} \frac{1}{N_{\Omega_j}} & & \left[
    \sum_{i=1}^{N_{\Omega_j}} |\Delta u_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \Delta u_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})|^2 
    + \sum_{i=1}^{N_{\Omega_j}} |\Delta v_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \Delta v_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})|^2
    \right] ,
\end{eqnarray} $$

Here $\Delta = \partial_{xx} + \partial_{yy}$ is the Laplacian operator. Then, the combined continuity loss term 
for XPINNs to infer ice viscosity becomes 

$$ \begin{equation}
    \mathcal{L_c} = \mathcal{L_{c0}} + \mathcal{L_{c1}} + \mathcal{L_{c2}}.
\end{equation} $$

Because the continuity of neural network prediction across the interface is important to ensure the uniqueness of the 
solutions, the importance of these loss terms in the cost function should be the same as the data loss $\mathcal{L}_{d}$. 
Thus, the loss weight for the continuity loss term $\mathcal{L}$ is set to be 1 by default. Therefore, the total cost 
function of XPINNs for inferring **isotropic** viscosity reads,

$$ \begin{equation}
    \mathcal{L} = \mathcal{L_d} + \mathcal{L_e} (\gamma_e, \gamma_b) + \mathcal{L_c} ,  \\ 
\end{equation} $$

and for inferring **anisotropic** viscosity gives,

$$ \begin{equation}
    \mathcal{L} = \mathcal{L_d} + \mathcal{L_e} (\gamma_e, \gamma_b) + \mathcal{L}_{reg}(\gamma_g)+\mathcal{L}_c
\end{equation} $$

where $\mathcal{L_d}$ and $\mathcal{L_e}$ are the data loss and equation loss as defined [here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/source/Isotropic.md). 
$\gamma_e$ and $\gamma_b$ are the weighting pre-factors for the equation and boundary condition loss. An extra
regularization loss term $\mathcal{L}_{reg}$ is required to infer anisotropic viscosity. The principle of this loss
terms and the appropriate value of its weight $\gamma_g$ are detailed described in 
[here](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/source/Anisotropic.md).
