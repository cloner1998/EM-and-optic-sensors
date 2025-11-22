# main.py
"""
Example runner to show a canonical EM + optical calculation.
"""

import numpy as np
from dataclass.emDataClass import EmSensorsParameters
from model.emModel import EMSensorsModel
from dataclass.opticDataClass import OpticSensorsParameters
from model.opticModel import OpticalSensorModel

def example_em():
    params = EmSensorsParameters(
        P_t=100.0,           # W
        G_t=10.0,            # linear
        G_r=10.0,
        frequency=1e9,       # 1 GHz
        P_r_min=1e-12,       # W
        L_system=1.0,
        alpha=0.0,
        D=0.5,
        SNR_db=10.0,
        cross_section=1.0,
        sensor_type="Active"
    )
    model = EMSensorsModel(params)
    R = model.effective_range()

    print(f"Estimated max active detection range: {R:.1f} m")

def example_optic():
    params = OpticSensorsParameters(
        I_0=1.0,
        I_min=1e-6,
        wavelength=550e-9,
        beta=0.1,
        D=0.05,
        pixel_pitch=5e-6,
        focal_length=0.05,
        SNR_db=30.0
    )
    model = OpticalSensorModel(params)
    R = model.effective_range()
    print(f"Optical effective range (Beer-Lambert): {R:.1f} m")
    print(f"Optical imaging sigma_theta (rad): {model.angle_error():.3e}")

if __name__ == '__main__':
    print("initializing ...")
    example_em()
    example_optic()
