import jax.numpy as jnp

def loss_create(predf, eqn_all, scale, lw):
    ''' a function factory to create the loss function based on given info
    :param predf: neural network function for solutions
    :param eqn_all: governing equation and boundary conditions
    :return: a loss function (callable)
    '''

    # define the mean squared error
    def _ms_error(diff):
        return jnp.mean(jnp.square(diff), axis=0)

    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # loss function used for the PINN training
    def loss_fun(params, data):
        # update the loss_ref
        loss_ref = loss_fun.lref
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the data of normalization condition
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:3]

        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_err = _ms_error(u_pred - u_smp)
        # calculate the mean squared root error of equation
        eqn_err = _ms_error(f_pred)
        bd_err = _ms_error(f_bd)

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)

        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd) / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                data_err, eqn_err, bd_err])
        return loss, loss_info

    loss_fun.lref = 1.
    return loss_fun
