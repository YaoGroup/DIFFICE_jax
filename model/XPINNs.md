# Extended physics-informed neural networks (XPINNs) for inferring effective viscosity of large ice shelves.

Large ice shelves surrounding the Antarctica, such as Ross and Ronne Ice Shelves, contain many local structural provinces,
including suture zones, margin shear zones, and fast-flowing zones, causing their physical properties, such as viscosity,
with dense local variations. However, neural networks training exhibits spectral biases [spectra biases](https://proceedings.mlr.press/v97/rahaman19a/rahaman19a.pdf)
for which, the neural networks tend to learn low-frequency information of the target function, while omitting the high-frequency one.
Considering a large ice shelf as a singje domain, those local variations thus become high-frequency information that
is hard to capture by a single neural network. 

To resolve this issuse, the extended-PINNs (XPINNs) separate any large domain into several sub-regions (see the figure)
and arrange different networks for them. In that case, each network only need to capture the physical variables within 
a small local region, largely increase the capacity for predicting the local variation of different physical variables.

Compared to regular PINN training, the key difference of XPINN training is to impose the continuity conditions at the
interface (blue line in the figure) between consecutive sub-regions that ensure the neural network predictions to be
continuous and smooth across the sub-regions. The continuity condition is also essential for the uniqueness of the 
inferred quantities from PINNs.

<p align="center">
    <img src="xpinns.png" alt="results" width="90%">
</p>

## Loss function for XPINNs

To ensure that the neural networks prediction of two consecutive sub-regions remain continuous at their interface, 
Several continuity loss terms needs to be added in the loss function. The first continuity loss term is given as

$$ \begin{eqnarray}
    \mathcal{L_{c0}} &=& \sum_{j=1}^{N_s} \frac{1}{N_{\Omega_j}} \left[\sum_{i=1}^{N_{\Omega_j}} [u_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - u_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 + \sum_{i=1}^{N_{\Omega_j}} [v_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - v_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 + \right. \\
    & & \left.\sum_{i=1}^{N_{\Omega_j}} [h_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - h_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 + \sum_{i=1}^{N_{\Omega_j}} [\mu_{j}^{(+)}({\bf \hat{x_{\Omega_j}}}^{(i)}) - \mu_j^{(-)}({\bf \hat{x_{\Omega_j}}}^{(i)})]^2 \right]
\end{eqnarray} $$

where ${\bf \hat{x_{\Omega_j}}}=(\hat{x_{\Omega_j}},\ \hat{y_{\Omega_j}})$ are the normalized locations for the 
collocation points on the $j$-th interface $\Omega_j$ between two adjacent networks and $N_{\Omega_j}$ is their total
number. $q_{j}^{(+)}$ and $q_{j}^{(-)}$ represent the neural network predictions in the two sub-regions that intersect
at the $j$-th interface $\Omega_j$, where $q$ stands for $u$, $v$, $h$ or $\mu$. $N_s$ is the total number of the 
interfaces between two adjacent networks. 
