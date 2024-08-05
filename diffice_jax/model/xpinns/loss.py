import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import lax


# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


# take the nth power root with original sign
def nthrt(x, n):
    return jnp.sign(x) * jnp.abs(x) ** (1/n)


def sub_scale(scale):
    # define the global parameter
    rho = 917
    rho_w = 1030
    gd = 9.8 * (1 - rho / rho_w)  # gravitational acceleration
    # load the scale information
    dmean, drange = scale
    lx0, ly0, u0, v0 = drange[0:4]
    um, vm, h0 = dmean[2:5]
    # find the maximum velocity and length scale
    u0m = lax.max(u0, v0)
    l0m = lax.max(lx0, ly0)
    # calculate the scale of viscosity and strain rate
    mu0 = rho * gd * h0 * (l0m / u0m)
    du0 = u0m / l0m
    dh0 = h0 / l0m
    term0 = h0**2 / l0m
    return u0, v0, h0, mu0, du0, dh0, term0, um/u0, vm/v0


#%% loss for inferring isotropic viscosity

def loss_iso_create(solNN, eqn_all, scale, idxgall, lw):
    ''' a function factory to create the loss function for isotropic analysis
    :param solNN: neural network function for solutions and its derivative [tuple(callable, callable)]
    :param eqn_all: include governing equation and boundary equation of SSA [tuple(callable, callable)]
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # obtain the viscosity and strain rate scale in each sub-region
    all_info = jnp.array(tree_map(lambda x: sub_scale(scale[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.mean(scale_info, axis=0)   # To do: check whether jnp.min or jnp.mean better
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    uvh0 = jnp.hstack([u0, v0, h0])
    um, vm = jnp.split(mean_nm, 2, axis=1)

    # create the loss constraint for each sub-regions
    def loss_sub(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)
        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]

        # load the position and weight of collocation points
        x_col = data['col'][0][idx]
        x_bd = data['bd'][0][idx]
        nn_bd = data['bd'][1][idx]

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]

        # calculate the residue of equation
        f_pred = gov_eqn(net, x_col, scale[idx])[0]
        f_bd = front_eqn(net, x_bd, nn_bd, scale[idx])[0]

        # calculate the mean squared error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err)) * uvh0[idx]
        # calculate the mean squared error of equation
        eqn_err = ms_error(f_pred) * term0[idx]
        bd_err = ms_error(f_bd) * h0[idx]
        # group all the error for output
        err_all = jnp.hstack([data_err, eqn_err, bd_err])
        return err_all

    # create the continuation loss constraint at the interface of adjacent subregions
    def loss_match(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)
        fgovterm = lambda x, id: gov_eqn(lambda x: net(x, id), x, scale[id])[1]
        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        """C0 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx]
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx]
        h_md1 = (U_md1[:, 2:3]) * h0[idx]
        mu_md1 = (U_md1[:, 3:4]) * mu0[idx]
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2 * jnp.log(mu_md1)])
        # obtain the variable in sub-region 2 at the interface
        U_md2 = net(x_md[:, 2:4], idx + 1)
        u_md2 = (U_md2[:, 0:1] + um[idx + 1]) * u0[idx + 1]
        v_md2 = (U_md2[:, 1:2] + vm[idx + 1]) * v0[idx + 1]
        h_md2 = (U_md2[:, 2:3]) * h0[idx + 1]
        mu_md2 = (U_md2[:, 3:4]) * mu0[idx + 1]
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2 * jnp.log(mu_md2)])
        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)

        """C1 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx]
        dh_md1 = dU_md1[:, 4:6] * dh0[idx]
        dvars_md1 = jnp.hstack([duv_md1, dh_md1])
        # obtain the variable in sub-region 2 at the interface
        dU_md2 = gdnet(x_md[:, 2:4], idx + 1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx + 1]
        dh_md2 = dU_md2[:, 4:6] * dh0[idx + 1]
        dvars_md2 = jnp.hstack([duv_md2, dh_md2])
        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        """C2 stitching condition at the boundary"""
        # calculate equation residue in sub-region 1 at the interface
        term_md1 = fgovterm(x_md[:, 0:2], idx)[:, 0:-1] * term0[idx]
        # calculate equation residue in sub-region 2 at the interface
        term_md2 = fgovterm(x_md[:, 2:4], idx + 1)[:, 0:-1] * term0[idx + 1]
        # calculate the c2 error
        match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitching conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.5])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        reg_err_list = tree_map(lambda x: loss_sub(params, data, x), idxgall)
        reg_err = jnp.mean(jnp.array(reg_err_list), axis=0)
        # calculate the error at the matching boundary
        match_err_list = tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1])
        match_err = jnp.mean(jnp.array(match_err_list), axis=0)
        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        data_w = jnp.array([1., 1., 0.6])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        md_w = jnp.ones(match_err.shape[0])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, md_w])

        # calculate the overall data loss and equation loss
        loss_each = err_all * wgh_all
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(loss_each[0:3])
        loss_eqn = jnp.sum(loss_each[3:5])
        loss_bd = jnp.sum(loss_each[5:7])
        loss_md = jnp.sum(loss_each[7:])

        # loading the pre-saved loss parameter
        loss_ref = loss_fun.lref
        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + lw[2] * loss_md)
        # normalize the loss by the initial reference value
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md]), err_all])
        return loss_n, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lref = 1.0

    return loss_fun


#%% loss for inferring anisotropic viscosity

def loss_aniso_create(solNN, eqn_all, scale, idxgall, lw):
    ''' a function factory to create the loss function for anisotropic analysis
    :param solNN: neural network function for solutions and its derivative [tuple(callable, callable)]
    :param eqn_all: include governing equation and boundary equation of SSA [tuple(callable, callable)]
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # obtain the viscosity and strain rate scale in each sub-region
    all_info = jnp.array(tree_map(lambda x: sub_scale(scale[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.mean(scale_info, axis=0)   # To do: check whether jnp.min or jnp.mean better
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    uvh0 = jnp.hstack([u0, v0, h0])
    um, vm = jnp.split(mean_nm, 2, axis=1)

    # create the loss constraint for each sub-regions
    def loss_sub(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)
        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]

        # load the position and weight of collocation points
        x_col = data['col'][0][idx]
        x_bd = data['bd'][0][idx]
        nn_bd = data['bd'][1][idx]

        # calculate the gradient of phi at origin
        output = net(x_smp)
        u_pred = output[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]

        # calculate the residue of equation
        f_pred = gov_eqn(net, x_col, scale[idx])[0]
        f_bd = front_eqn(net, x_bd, nn_bd, scale[idx])[0]

        # calculate the mean squared error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err)) * uvh0[idx]
        # calculate the mean squared error of equation
        eqn_err = ms_error(f_pred) * term0[idx]
        bd_err = ms_error(f_bd) * h0[idx]
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred)) / 2) * mu0[idx]

        # group all the error for output
        err_all = jnp.hstack([data_err, eqn_err, bd_err, sp_err])
        return err_all

    # create the continuation loss constraint at the interface of adjacent subregions
    def loss_match(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)
        fgovterm = lambda x, id: gov_eqn(lambda x: net(x, id), x, scale[id])[1]
        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        """C0 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx]
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx]
        h_md1 = (U_md1[:, 2:3]) * h0[idx]
        mu_md1 = (U_md1[:, 3:5]) * mu0[idx]     # include both mu and eta
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2*jnp.log(mu_md1)])
        # obtain the variable in sub-region 2 at the interface
        U_md2 = net(x_md[:, 2:4], idx+1)
        u_md2 = (U_md2[:, 0:1] + um[idx+1]) * u0[idx+1]
        v_md2 = (U_md2[:, 1:2] + vm[idx+1]) * v0[idx+1]
        h_md2 = (U_md2[:, 2:3]) * h0[idx+1]
        mu_md2 = (U_md2[:, 3:5]) * mu0[idx+1]   # include both mu and eta
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2*jnp.log(mu_md2)])
        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)

        """C1 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx]
        dh_md1 = dU_md1[:, 4:6] * dh0[idx]
        dvars_md1 = jnp.hstack([duv_md1, dh_md1])
        # obtain the variable in sub-region 2 at the interface
        dU_md2 = gdnet(x_md[:, 2:4], idx+1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx+1]
        dh_md2 = dU_md2[:, 4:6] * dh0[idx+1]
        dvars_md2 = jnp.hstack([duv_md2, dh_md2])
        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        """C2 stitching condition at the boundary"""
        # calculate equation residue in sub-region 1 at the interface
        term_md1 = fgovterm(x_md[:, 0:2], idx)[:, 0:-1] * term0[idx]
        # calculate equation residue in sub-region 2 at the interface
        term_md2 = fgovterm(x_md[:, 2:4], idx+1)[:, 0:-1] * term0[idx+1]
        # calculate the c2 error
        match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitching conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.5])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        reg_err_list = tree_map(lambda x: loss_sub(params, data, x), idxgall)
        reg_err = jnp.mean(jnp.array(reg_err_list), axis=0)
        # calculate the error at the matching boundary
        match_err_list = tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1])
        match_err = jnp.mean(jnp.array(match_err_list), axis=0)
        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        data_w = jnp.array([1., 1., 0.6])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        sp_w = jnp.array([1.])
        md_w = jnp.ones(match_err.shape[0])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, sp_w, md_w])

        # modify the contribution of each loss term by their weights
        loss_each = err_all * wgh_all
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(loss_each[0:3])
        loss_eqn = jnp.sum(loss_each[3:5])
        loss_bd = jnp.sum(loss_each[5:7])
        loss_sp = jnp.sum(loss_each[7:8])
        loss_md = jnp.sum(loss_each[8:])

        # loading the pre-saved loss parameter
        loss_ref = loss_fun.lref
        # load the weight for the regularization loss
        wsp = loss_fun.wsp
        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + lw[2] * loss_md + wsp * loss_sp)
        # normalize the loss by the initial reference value
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md, loss_sp]), err_all])
        return loss_n, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lref = 1.0
    loss_fun.wsp = lw[3]
    return loss_fun