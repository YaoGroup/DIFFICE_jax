# Examples

This folder provides four example codes that use `DIFFICE_jax` package to assimilate remote-sensing velocity
and thickness data and infer effective ice viscosity under either **isotropic** or **anisotropic** assumption
via regular PINNs or extended-PINNs (XPINNs).

## `train_pinns_aniso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under anisotropic
assumption via regular PINNs. The code are computationally-efficient and accurate enough to study ice shelves
of size close or smaller than Amery or Larce C Ice Shelves.
