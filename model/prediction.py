import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import lax

def dataArrange(var, idxval, dsize):
    nanmat = jnp.empty(dsize)
    nanmat = nanmat.at[:].set(jnp.nan)
    var_1d = nanmat.flatten()[:, None]
    var_1d = var_1d.at[idxval].set(var)
    var_2d = jnp.reshape(var_1d, dsize)
    return var_2d

def extract_scale(scale_info):
    # define the global parameter
    rho = 917
    rho_w = 1030
    gd = 9.8 * (1 - rho / rho_w)  # gravitational acceleration
    # load the scale information
    dmean, drange = scale_info
    lx0, ly0, u0, v0 = drange[0:4]
    lxm, lym, um, vm = dmean[0:4]
    h0 = dmean[4]
    # find the maximum velocity and length scale
    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    # calculate the scale of viscosity and strain rate
    mu0 = rho * gd * h0 * (l0m / u0m)
    str0 = u0m/l0m
    term0 = rho * gd * h0 ** 2 / l0m
    # group characteristic scales for all different variables
    scale = dict(lx0=lx0, ly0=ly0, u0=u0, v0=v0, h0=h0,
                 lxm=lxm, lym=lym, um=um, vm=vm,
                 mu0=mu0, str0=str0, term0=term0)
    return scale


def predict(func_all, data_all):
    idxval, dsize = data_all[4][-2:]
    data_norm = data_all[4][2]
    scale = data_all[4][0:2]
    x_star, y_star, u_star, v_star, h_star = data_norm[0:5]
    xout, yout = data_all[4][4][0:2]
    dsout = xout.shape
    xo = xout.flatten()[:, None]
    yo = yout.flatten()[:, None]
    idxout = jnp.where(~jnp.isnan(xo))[0]
    x_pred = jnp.hstack([xo, yo])[idxout]

    # extract the function of solution and equation residue
    [f_u, f_gu, f_eqn] = func_all

    # extract the scale for different variables
    varscl = extract_scale(scale)

    # separate the array for the memory limit
    nsp = 4
    # separate the array for the memory limit
    x_psp = jnp.array_split(x_pred, nsp)
    idxsp = jnp.arange(nsp).tolist()
    # calculate the data for each sub-group
    u_list = tree_map(lambda x: f_u(x_psp[x]), idxsp)
    du_list = tree_map(lambda x: f_gu(x_psp[x]), idxsp)
    eqnterm_list = tree_map(lambda x: f_eqn(x_psp[x]), idxsp)
    eqn_list = tree_map(lambda x: eqnterm_list[x][0], idxsp)
    term_list = tree_map(lambda x: eqnterm_list[x][1], idxsp)
    # combine the sub-group list into a long array
    uvhm = jnp.vstack(u_list)
    duvh = jnp.vstack(du_list)
    eqn = jnp.vstack(eqn_list)
    term = jnp.vstack(term_list)

    # convert to 2D dimensional dataset
    x = dataArrange(x_star, idxval, dsize) * varscl['lx0'] + varscl['lxm']
    y = dataArrange(y_star, idxval, dsize) * varscl['ly0'] + varscl['lym']
    u_data = dataArrange(u_star, idxval, dsize) * varscl['u0'] + varscl['um']
    v_data = dataArrange(v_star, idxval, dsize) * varscl['v0'] + varscl['vm']
    h_data = dataArrange(h_star, idxval, dsize) * varscl['v0']

    # convert to 2D NN prediction
    u_p = dataArrange(uvhm[:, 0:1], idxout, dsout) * varscl['u0'] + varscl['um']
    v_p = dataArrange(uvhm[:, 1:2], idxout, dsout) * varscl['v0'] + varscl['vm']
    h_p = dataArrange(uvhm[:, 2:3], idxout, dsout) * varscl['h0']
    mu_p = dataArrange(uvhm[:, 3:4], idxout, dsout) * varscl['mu0']

    # convert to 2D derivative of prediction
    ux_p = dataArrange(duvh[:, 0:1], idxout, dsout) * varscl['u0']/varscl['lx0']
    uy_p = dataArrange(duvh[:, 1:2], idxout, dsout) * varscl['u0']/varscl['ly0']
    vx_p = dataArrange(duvh[:, 2:3], idxout, dsout) * varscl['v0']/varscl['lx0']
    vy_p = dataArrange(duvh[:, 3:4], idxout, dsout) * varscl['v0']/varscl['ly0']
    hx_p = dataArrange(duvh[:, 4:5], idxout, dsout) * varscl['h0']/varscl['lx0']
    hy_p = dataArrange(duvh[:, 5:6], idxout, dsout) * varscl['h0']/varscl['ly0']

    # convert to 2D equation residue
    e1 = dataArrange(eqn[:, 0:1], idxout, dsout) * varscl['term0']
    e2 = dataArrange(eqn[:, 1:2], idxout, dsout) * varscl['term0']

    # convert to 2D equation term value
    e11 = dataArrange(term[:, 0:1], idxout, dsout) * varscl['term0']
    e12 = dataArrange(term[:, 1:2], idxout, dsout) * varscl['term0']
    e13 = dataArrange(term[:, 2:3], idxout, dsout) * varscl['term0']
    e21 = dataArrange(term[:, 3:4], idxout, dsout) * varscl['term0']
    e22 = dataArrange(term[:, 4:5], idxout, dsout) * varscl['term0']
    e23 = dataArrange(term[:, 5:6], idxout, dsout) * varscl['term0']
    strate = dataArrange(term[:, -1:], idxout, dsout) * varscl['str0']

    # group all the variables
    results = {"x": x, "y": y, "u": u_p, "v": v_p, "h": h_p, "mu": mu_p,
            "u_g": u_data, "v_g": v_data, "h_g": h_data,
            "u_x": ux_p, "u_y": uy_p, "v_x": vx_p, "v_y": vy_p,
            "h_x": hx_p, "h_y": hy_p, "str": strate,
            "e11": e11, "e12": e12, "e13": e13,
            "e21": e21, "e22": e22, "e23": e23,
            "e1": e1, "e2": e2, "scale": varscl}

    return results
