from Examples import *
import pytest

mat_model_list = ['nh', 'mr', 'yeoh','ls','mn','expI1','goh','Holzapfel','hgo','hy','volPenalty','polyI4','ArrudaBoyce','Gent','splineI1','splineI1I4','StructModel']
samples_list = [UniaxialExtension,PlanarBiaxialExtension,TubeInflation, LinearSpring]

def test_mat_creation():
    #Test creating all individual material models
    for mname in mat_model_list:
        output = MatModel(mname)
        mm = output.models
        mm[0].fiber_dirs = [np.array([1,0,0]),np.array([0.5,0.5,0])]
        assert isinstance(output, MatModel)
        assert len(output.models) == 1
        assert len(output.models[0].fiber_dirs) == 2 
