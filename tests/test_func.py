import pymecht as pmt
import pytest
import numpy as np

def test_arbitrary_MatModel_I1():
    model = pmt.ARB('mu/2.*(I1-3.)','mu=1.','mu=0.01','mu=10.')
    mat = pmt.MatModel(model)
    mat2 = pmt.MatModel('NH')
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
    
def test_arbitrary_MatModel_I2():
    model = pmt.ARB('c1*(I1-3.) + c2*(I2-3.)','c1=1., c2=1.','c1=0.0001, c2=0.','c1=100., c2=100.')
    mat = pmt.MatModel(model)
    mat2 = pmt.MatModel('MR')
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
    
def test_arbitrary_MatModel_J():
    model = pmt.ARB('kappa/2.*(J-1.)**2','kappa=1.','kappa=1.','kappa=1.')
    mat = pmt.MatModel(model)
    mat2 = pmt.MatModel('volPenalty')
    assert (mat.stress(random_F)-mat2.stress(random_F)) == pytest.approx(np.zeros((3,3)))
