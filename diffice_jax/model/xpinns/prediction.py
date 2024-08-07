import sys
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax import lax
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from equation.eqn_iso import vectgrad


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


def net_output(func_all, data_norm, scale, idx, nsp=4):
    # obtained the normalized dataset
    x_star, y_star, u_star, v_star, xh_star, yh_star, h_star = data_norm
    # set the output position based on the original velocity data
    x_pred = jnp.hstack([x_star, y_star])
    # set the output position based on the original thickness data
    xh_pred = jnp.hstack([xh_star, yh_star])

    # extract the function of solution and equation residue
    [f_u_idx, gov_eqn] = func_all
    f_u = lambda x: f_u_idx(x, idx)
    f_gu = lambda x: vectgrad(f_u, x)[0][:, 0:6]
    f_eqn = lambda x: gov_eqn(f_u, x, scale)

    # calculate the network output at the original velocity-data positions
    uvhm = f_u(x_pred)
    # calculate the network output at the original thickness-data positions
    h2 = f_u(xh_pred)[:, 2:3]

    # separate input into different partition to avoid GPU memory limit
    x_psp = jnp.array_split(x_pred, nsp)
    idxsp = jnp.arange(nsp).tolist()
    # calculate the derivative of network output at the velocity-data positions
    du_list = tree_map(lambda x: f_gu(x_psp[x]), idxsp)
    # calculate the associated equation residue of the trained network
    eqnterm_list = tree_map(lambda x: f_eqn(x_psp[x]), idxsp)
    eqn_list = tree_map(lambda x: eqnterm_list[x][0], idxsp)
    term_list = tree_map(lambda x: eqnterm_list[x][1], idxsp)
    # combine the sub-group list into a single array
    duvh = jnp.vstack(du_list)
    eqn = jnp.vstack(eqn_list)
    term = jnp.vstack(term_list)

    return uvhm, h2, duvh, eqn, term


def redimensionalize(output, data_norm, data_info, idxgall, aniso):
    """ re-shape all the output variables into the same grid with the original data
        and re-dimensionalize the output variables into their original (SI) unit
    :param output: all output variables in each sub-region [list]
    :param data_norm: normalized velocity and thickness data in each sub-region [list]
    :param data_info: information required to re-shape and re-dimensionalize the output variables
    :param idxgall: index of the sub-regions
    :param aniso: whether the result is for anisotropic anslysis
    :return: stitched variables in the whole domain [Array]
    """

    # extract all the information for re-shaping and re-dimensionalization
    idxval, idxval_h, dsize, dsize_h, scale = data_info
    # extract the scale information for each variable
    varscl = tree_map(lambda x: extract_scale(scale[x]), idxgall)

    # convert to 2D original velocity dataset
    x = tree_map(lambda x: dataArrange(
        data_norm[x][0], idxval[x], dsize[x]) * varscl[x]['lx0'] + varscl[x]['lxm'], idxgall)
    y = tree_map(lambda x: dataArrange(
        data_norm[x][1], idxval[x], dsize[x]) * varscl[x]['ly0'] + varscl[x]['lym'], idxgall)
    u_data = tree_map(lambda x: dataArrange(
             data_norm[x][2], idxval[x], dsize[x]) * varscl[x]['u0'] + varscl[x]['um'], idxgall)
    v_data = tree_map(lambda x: dataArrange(
             data_norm[x][3], idxval[x], dsize[x]) * varscl[x]['v0'] + varscl[x]['vm'], idxgall)

    # convert to 2D original thickness dataset
    x_h = tree_map(lambda x: dataArrange(
          data_norm[x][4], idxval_h[x], dsize_h[x]) * varscl[x]['lx0'] + varscl[x]['lxm'], idxgall)
    y_h = tree_map(lambda x: dataArrange(
          data_norm[x][5], idxval_h[x], dsize_h[x]) * varscl[x]['ly0'] + varscl[x]['lym'], idxgall)
    h_data = tree_map(lambda x: dataArrange(
             data_norm[x][6], idxval_h[x], dsize_h[x]) * varscl[x]['h0'], idxgall)

    # convert to 2D NN prediction
    u_p = tree_map(lambda x: dataArrange(
          output[x][0][:, 0:1], idxval[x], dsize[x]) * varscl[x]['u0'] + varscl[x]['um'], idxgall)
    v_p = tree_map(lambda x: dataArrange(
          output[x][0][:, 1:2], idxval[x], dsize[x]) * varscl[x]['v0'] + varscl[x]['vm'], idxgall)
    h_p = tree_map(lambda x: dataArrange(
          output[x][0][:, 2:3], idxval[x], dsize[x]) * varscl[x]['h0'], idxgall)
    h_p2 = tree_map(lambda x: dataArrange(
           output[x][1], idxval_h[x], dsize_h[x]) * varscl[x]['h0'], idxgall)
    mu_p = tree_map(lambda x: dataArrange(
           output[x][0][:, 3:4], idxval[x], dsize[x]) * varscl[x]['mu0'], idxgall)

    # convert to 2D derivative of prediction
    ux_p = tree_map(lambda x: dataArrange(
           output[x][2][:, 0:1], idxval[x], dsize[x]) * varscl[x]['u0']/ varscl[x]['lx0'], idxgall)
    uy_p = tree_map(lambda x: dataArrange(
           output[x][2][:, 1:2], idxval[x], dsize[x]) * varscl[x]['u0']/ varscl[x]['ly0'], idxgall)
    vx_p = tree_map(lambda x: dataArrange(
           output[x][2][:, 2:3], idxval[x], dsize[x]) * varscl[x]['v0']/ varscl[x]['lx0'], idxgall)
    vy_p = tree_map(lambda x: dataArrange(
           output[x][2][:, 3:4], idxval[x], dsize[x]) * varscl[x]['v0']/ varscl[x]['ly0'], idxgall)
    hx_p = tree_map(lambda x: dataArrange(
           output[x][2][:, 4:5], idxval[x], dsize[x]) * varscl[x]['h0']/ varscl[x]['lx0'], idxgall)
    hy_p = tree_map(lambda x: dataArrange(
           output[x][2][:, 5:6], idxval[x], dsize[x]) * varscl[x]['h0']/ varscl[x]['ly0'], idxgall)

    # convert to 2D equation residue
    e1 = tree_map(lambda x: dataArrange(
         output[x][3][:, 0:1], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    e2 = tree_map(lambda x: dataArrange(
         output[x][3][:, 1:2], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)

    # convert to 2D equation term value
    e11 = tree_map(lambda x: dataArrange(
          output[x][4][:, 0:1], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    e12 = tree_map(lambda x: dataArrange(
          output[x][4][:, 1:2], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    e13 = tree_map(lambda x: dataArrange(
          output[x][4][:, 2:3], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    e21 = tree_map(lambda x: dataArrange(
          output[x][4][:, 3:4], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    e22 = tree_map(lambda x: dataArrange(
          output[x][4][:, 4:5], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    e23 = tree_map(lambda x: dataArrange(
          output[x][4][:, 5:6], idxval[x], dsize[x]) * varscl[x]['term0'], idxgall)
    strate = tree_map(lambda x: dataArrange(
          output[x][4][:, -1:], idxval[x], dsize[x]) * varscl[x]['str0'], idxgall)

    # output variable calculated in the grid of original velocity data
    varsub = [x, y, u_data, v_data, u_p, v_p, h_p,
              ux_p, uy_p, vx_p, vy_p, hx_p, hy_p, strate,
              e1, e2, e11, e12, e13, e21, e22, e23, mu_p]
    # create a index list for each of output
    idxvars = jnp.arange(len(varsub)).tolist()

    # output variable calculated in the grid of original thickness data
    varsub_h = [x_h, y_h, h_data, h_p2]
    # create a index list for each of output
    idxvars_h = jnp.arange(len(varsub_h)).tolist()

    # for the anisotropic case
    if aniso:
        # re-shape and re-dimensionalize the second viscosity components
        eta_p = tree_map(lambda x: dataArrange(
            output[x][0][:, 4:5], idxval[x], dsize[x]) * varscl[x]['mu0'], idxgall)
        # add the second viscosity components to the list of output variables
        varsub.append(eta_p)

    return varsub, varsub_h, idxvars, idxvars_h


def stitch(vars_sub, idxcrop, fullsize, idxgall):
    """stitch the variable in each sub-region into the whole domain.
    :param vars_sub: value of the variables in each subdomain [list]
    :param idxcrop: relative position of each sub-region in the whole domain [Array]
    :param fullsize: size of the whole domain matrix  [Array: (2,)]
    :return: stitched variables in the whole domain [Array]
    """
    # create the nan matrix with the whole domain size (dsize)
    nanmat = jnp.empty(fullsize)
    nanmat = nanmat.at[:].set(jnp.nan)
    # patch each of the subregion to the whole domain individually
    vars_patch = jnp.array(tree_map(
        lambda x: nanmat.at[idxcrop[x, 2]-1:idxcrop[x, 3], idxcrop[x, 0]-1:idxcrop[x, 1]].set(vars_sub[x]), idxgall))
    # merge all the subregion in the whole domain into one matrix
    vars_merge = jnp.nanmean(vars_patch, axis=0)
    return vars_merge


def predict(func_all, data_all, posi_all, idxcrop_all, idxgall, aniso=False):
    # %% calculate the output data

    # extract the non-nan index of the original dataset
    idxval = tree_map(lambda x: data_all[x][4][-2][0], idxgall)
    idxval_h = tree_map(lambda x: data_all[x][4][-2][1], idxgall)
    # extract the 2D shape of the original dataset
    dsize = tree_map(lambda x: data_all[x][4][-1][0], idxgall)
    dsize_h = tree_map(lambda x: data_all[x][4][-1][1], idxgall)
    # extract the scale for different variables
    scale = tree_map(lambda x: data_all[x][4][0:2], idxgall)
    # group all the above information
    data_info = (idxval, idxval_h, dsize, dsize_h, scale)

    # extract the position matrix for the whole domain of ice shelf.
    Xe, Ye, Xe_h, Ye_h = posi_all
    fullsize = Xe.shape
    fullsize_h = Xe_h.shape

    # extract the idxcrop for both velocity and thickness data
    idxcrop, idxcrop_h = idxcrop_all

    # obtained the normalized dataset
    data_norm = tree_map(lambda x: data_all[x][4][2], idxgall)

    # calculate the trained network output and associated equation residue at given positions
    output = tree_map(lambda x: net_output(func_all, data_norm[x], scale[x], x), idxgall)

    # re-shape and re-dimensonalize each output variable into their original shape and unit
    varsub, varsub_h, idxvars, idxvars_h = redimensionalize(output, data_norm, data_info, idxgall, aniso)

    # stitch the output variables calculated in the grid of original velocity data into one matrix
    results = tree_map(lambda x: stitch(varsub[x], idxcrop, fullsize, idxgall), idxvars)
    # stitch the output variables calculated in the grid of original thickness data into one matrix
    results_h = tree_map(lambda x: stitch(varsub_h[x], idxcrop_h, fullsize_h, idxgall), idxvars_h)

    # check whether the sub-regions merge correctly
    merge_check1 = jnp.nanmean(jnp.abs(results[0]-Xe)) == 0
    merge_check2 = jnp.nanmean(jnp.abs(results[1]-Ye)) == 0
    merge_check3 = jnp.nanmean(jnp.abs(results_h[0]-Xe_h)) == 0
    merge_check4 = jnp.nanmean(jnp.abs(results_h[1]-Ye_h)) == 0
    merge_check = merge_check1 & merge_check2 & merge_check3 & merge_check4
    # if not correct, stop the code and show the error message
    assert merge_check, "Sub-region merges fails. Please check the code."

    # group all the variables
    outvars = {"x": results[0], "y": results[1], "u_g": results[2], "v_g": results[3],
               "u": results[4], "v": results[5], "h": results[6],
               "u_x": results[7], "u_y": results[8], "v_x": results[9], "v_y": results[10],
               "h_x": results[11], "h_y": results[12], "str": results[13],
               "e1": results[14], "e2": results[15],
               "e11": results[16], "e12": results[17], "e13": results[18],
               "e21": results[19], "e22": results[20], "e23": results[21],
               "mu": results[22],
               "x_h": results_h[0], "y_h": results_h[1], "h_g": results_h[2], "h2": results_h[3]}
    if aniso:
        outvars['eta'] = results[-1]

    return outvars
