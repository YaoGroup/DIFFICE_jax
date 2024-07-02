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
