import diffice_jax as djax
import pytest
from jax import random
import jax.numpy as jnp
from scipy.io import loadmat
import math
import os

seed = 1234
key = random.PRNGKey(seed)
keys = random.split(key, 4)

#%% check the network weight initialization
@pytest.mark.parametrize("nh, nu, nsub, nblock", [
    (4, 25, 4, 5),
    (3, 23, 3, 4),
    (6, 26, 5, 7),
])

def test_initialization(nh, nu, nsub, nblock):
    # test the initialization of regular PINNs
    trained_params = djax.init_pinn(keys[0], nh, nu)
    assert isinstance(trained_params, list)
    assert (len(trained_params)) == pytest.approx(2)
    assert (len(trained_params[0])) == pytest.approx(nblock)
    assert (len(trained_params[1])) == pytest.approx(nblock)

    # test the initialization of extended PINNs
    trained_params2 = djax.init_xpinn(keys[0], nh, nu, nsub)
    assert len(trained_params2) == pytest.approx(2)
    assert isinstance(trained_params2, dict)
    assert "net_u" in trained_params2
    assert "net_mu" in trained_params2
    assert len(trained_params2["net_u"]) == nsub
    assert len(trained_params2["net_mu"]) == nsub


#%%
"""check the normalization of observational data"""

# create the dataset filename
filename = 'data_pinns_test.mat'
filepath = os.path.join(os.path.dirname(__file__), filename)
filename2 = 'data_xpinns_test.mat'
filepath2 = os.path.join(os.path.dirname(__file__), filename2)

# load the datafile
rawdata_pinn = loadmat(filepath)
rawdata_xpinn = loadmat(filepath2)
data_pinn = djax.normdata_pinn(rawdata_pinn)
data_xpinn = djax.normdata_xpinn(rawdata_xpinn)
n_reg = len(rawdata_xpinn['xd'][0])


# Fixture with assert
@pytest.fixture
def check_normalization():
    def _check(data_each):
        # test the data normalization for regular PINNs
        assert len(data_each[0]) == 2
        assert len(data_each[1]) == 2
        assert data_each[2].shape == data_each[3].shape
        assert len(data_each[4]) == 6
        assert data_each[4][0].shape == (5,)
        assert data_each[4][1].shape == (5,)
        assert len(data_each[4][2]) == 7
        assert len(data_each[4][3]) == 7
        assert data_each[0][0].shape[0] == data_each[4][4][0].shape[0]
        assert data_each[0][1].shape[0] == data_each[4][4][1].shape[0]
        assert math.prod(data_each[4][5][0]) >= data_each[4][4][0].shape[0]
        assert math.prod(data_each[4][5][1]) >= data_each[4][4][1].shape[0]
    return _check

def test_normalization(check_normalization):
    # test the data normalization for regular PINNs
    assert len(data_pinn) == 5
    check_normalization(data_pinn)

    # test the data normalization for extended PINNs
    assert len(data_xpinn) == 4
    assert len(data_xpinn[0]) == n_reg
    check_normalization(data_xpinn[0][0])
    assert len(data_xpinn[2]) == 4
    assert len(data_xpinn[3]) == 2
    assert len(data_xpinn[3][0]) == n_reg
    assert len(data_xpinn[3][1]) == n_reg


#%%
"""check the neural network formation"""

@pytest.mark.parametrize("nd, nsub", [
    (100, 2),
    (200, 3),
    (300, 4),
])

def test_network_create(nd, nsub):
    x = jnp.ones((nd, 2))
    trained_params = djax.init_pinn(keys[0], 4, 25)
    predf_pinn = djax.solu_pinn()
    assert callable(predf_pinn)
    output = predf_pinn(trained_params, x)
    assert output.shape == pytest.approx((nd, 4))

    trained_params2 = djax.init_xpinn(keys[0], 4, 25, nsub)
    idx_rand = random.choice(keys[1], nsub)
    idx = min([n_reg, idx_rand])
    scale = data_xpinn[0][idx][4][0:2]
    predf_xpinn = djax.solu_xpinn(scale)
    assert len(predf_xpinn) == 2
    assert callable(predf_xpinn[0])
    assert callable(predf_xpinn[1])
    output2 = predf_xpinn[0](trained_params2, x, idx)
    assert output2.shape == pytest.approx((nd, 4))




