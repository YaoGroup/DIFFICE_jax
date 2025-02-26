[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/tutorial/colab/train_syndata.ipynb)

# Overview

Here, we provide a tutorial example for inferring the effective viscosity $\mu$ of ice 
shelves using synthetic data of ice velocity and thickness via physics-informed neural networks 
(PINNs). Both the simulation code for synthetic data generation and the PINN code for viscosity
inversion are included. All codes are well-documented for easy understanding. Additionally, we
have provided a [Colab Notebook](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/tutorial/colab/train_syndata.ipynb) that allows users to run the code online in Google Colab.

<br />

# Forward problem setup
Considering the floating ice moving in a given domain, the synthetic data of ice velocity 
and thickness can be calculated by numerically solving the Shallow-shelf Approximation (SSA)
equations and the steady mass conservation equation with a given viscosity, which read

$$\begin{array}{l}
\displaystyle \frac{\partial} {\partial x}\left(4 \mu h \frac{\partial  u}{\partial x} + 2\mu h \frac{\partial  v}{\partial y}  \right) 
	+ \frac{\partial} {\partial y} \left( \mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x}  \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial x} \qquad \text{(SSA)} \cr
\displaystyle  \frac{\partial} {\partial x} \left(\mu h\frac{\partial  u}{\partial y} + \mu h\frac{\partial v}{\partial x} \right)
	+ \frac{\partial} {\partial y} \left( 2\mu h\frac{\partial u}{\partial x} + 4 \mu h\frac{\partial v}{\partial y} \right)  
= \rho g \left(1-\frac{\rho}{\rho_w}\right)h\frac{\partial h}{\partial y} \qquad \text{(SSA)} \cr
\end{array}$$

$$ \qquad \nabla \cdot (hu) = \frac{\partial (hu)} {\partial x} + \frac{\partial (hv)} {\partial y} = \dot{a} - \dot{b} \qquad \qquad \text{(Steady mass conserv.)} $$

where $(u, v)$, $h$ and $\mu$ are the velocity vector, thickness and viscosity of the floating 
ice, respectively.  $\dot{a}$ and $\dot{b}$  are the snow accumulation rate and basel melting rate,
respectively. For the tutorial example provided in this folder, we consider floating 
ice moving in a confined rectangular channel. For simplicity, we assume that both $\dot{a}$ and $\dot{b}$
are equal to 0. The domain size and the associated boundary conditions for the ice flow are listed 
in the figure below. To make the example realistic for actual ice-shelf flow, we set the velocity 
scale to $u_0 = 1$ $\mathrm{km/yr}$ $= 3.17 \times 10^{-5}$ $\mathrm{m/s}$ and the thickness 
scale to $h_0 = 500$ $\mathrm{m}$.

<p align="center">
    <img alt="boundary conditions" width="100%" src="https://github.com/YaoGroup/DIFFICE_jax/raw/main/docs/figure/syndata_cond.png"> 
</p>

Besides the governing equations and the boundary condition, a **known** viscosity profile $\mu(x,y)$ 
is required to generate the synthetic data of ice velocity and thickness. For the tutorial example,
the viscosity profile is given as


$$ \begin{equation}
  \mu(x,y) = \mu_0 \left[1-\frac{1}{2} \cos \left(2\pi \frac{y}{L_y}\right)\right] \left(\frac{2}{3} + \frac{x}{3L_x}\right)
\end{equation} $$

where we set the viscosity scale to $\mu_0 = 5 \times 10^{13}$ (Pa $\cdot$ s) to match the 
magnitude of actual ice-shelf viscosity. With all the necessary information, we can generate
the synthetic data by numerically solving the govnering equation. For simplicity, we 
solve the equations using **COMSOL Multiphysics**, which, we believe, provides a intuitive 
user interface to set up the forward problem and conduct the calculation. 

<br />

# Code description
### `tutorial/COMSOL/IceShelf2D_forward.mph`  

A COMSOL file `.mph` in the `COMSOL` folder solves the governing equations with the boundary 
conditions and given viscosity as described above. Users need to have the basic COMSOL software 
(no extra Module required) with version >= 5.6 to open the file. Users are free to change the 
domain size, geometry, boundary conditions, and given viscosity profile in the COMSOL file to 
create different synthetic data. The provided COMSOL file can export the synthetic data in a `.txt` 
format by default. The `SynData_exp1.txt` is the data file exported from the current COMSOL file.

<br />

### `tutorial/COMSOL/txt2mat.m` 

A MATLAB script that converts the raw data file (`.txt`) exported from COMSOL into MATLAB data 
format (`.mat`). The synthetic data in `.mat` format are also organized in the way that allow them
to be loaded into the Python code for the PINN training. The current `SynData_exp1.mat` in the `COMSOL`
folder is the MATLAB data file converted from the `SynData_exp1.txt` raw data file. Additionally,
we recognize that the MATLAB data format (`.mat`) is convenient for users to observe the synthetic
data in MATLAB using simple commands.

```matlab
load('SynData_exp1.mat')
figure; surf(xd, yd, ud);  % surface plot of the velocity component u
shading interp;

figure; surf(xd_h, yd_h, hd);  % surface plot of the thickness h (at different grids)
shading interp;
```

<br />


### `tutorial/train_syndata.py`

The main Python script conducts the PINN training to infer ice viscosity from the synthetic data.
This script is intended to be run on a local machine or a cluster. Users should select the 
synthetic data file and set the hyper-parameters for the training. After the training is complete,
the script will automatically save the trained network weights and biases in `.pkl` format, and the
predictions of the solution and equation residue at high-resolution grids in `.mat` format. Both 
files will be stored in the `Results` subfolder, which will be automatically generated if it does 
not already exist. The current hyper-parameters set in the script allow the viscosity inversion 
from the example data `SynData_exp1.mat` to achieve high accuracy.

<br />


### `tutorial/train_syndata.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/tutorial/colab/train_syndata.ipynb)

The Colab notebook, similar to the script `pinn_syndata.py` that can conduct the PINN training to infer ice 
viscosity from the synthetic data. The user can run the notebook directly in Google Colab online without any 
need to install python environments and library on a local machine. Different from the script `pinn_syndata.py`,
after the training, the notebook plots the trained networks for the data assimilation and viscosity 
inversion, and compare them directly with the synthetic data and the given viscosity profile.

<br />


# Results

The figure below shows the trained results of PINNs for the synthetic data provided in this folder. The 
trained networks for ice velocity and thickness match well with the synthetic data, and the inferred 
viscosity shows a good agreement with the given viscosity profile with a relative error less than 1%.
This results were obtained after **100k** iterations of training using **Adams** optimizer, followed by another **100K** 
iterations of training using **L-BFGS** optimizer.

<p align="center">
    <img alt="results" width="90%" src="https://github.com/YaoGroup/DIFFICE_jax/raw/main/docs/figure/results.png"> 
</p>
