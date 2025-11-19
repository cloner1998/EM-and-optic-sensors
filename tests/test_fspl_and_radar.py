import numpy as np
import pytest
from dataclass.emDataClass import EmSensorsParameters
from model.emModel import EMSensorsModel

def test_fspl_behavior():
    params = EmSensorsParameters(P_t=1.0,G_t=1.0,G_r=1.0,frequency=3e8,P_r_min=1e-12,L_system=1.0,alpha=0.0,D=0.1,SNR_db=10.,cross_section=1.0)
    model = EMSensorsModel(params)
    fspl1 = model.free_space_path_loss_linear(1.0)
    fspl2 = model.free_space_path_loss_linear(2.0)
    assert fspl2 > fspl1