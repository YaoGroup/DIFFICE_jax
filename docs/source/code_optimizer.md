# Optimization methods

## Location: `diffice_jax/optimizer`

### `/optimization.py`

Providing provides two optimization methods:

[**Adam**](https://arxiv.org/pdf/1412.6980): a first-order gradient-based optimization method 
based on adaptive estimates of lower-order moments.

[**L-BFGS**](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf): Limited-memory BFGS method, a second-order quasi-Newton optimization method for
solving unconstrained nonlinear optimization problems, using a limited amount of computer memory.

A **stochastic training scheme** is applied to two optimizers, where the code randomizes both data samples and collocation points at regular interval during training 
to minimize the cheating effect.
