"""
Optical sensor model: effective range via Beer-Lambert, imaging angle error approximation.

Units:
 - distances in meters
 - irradiance I_0, I_min in W/m^2
 - wavelength in meters
 - beta extinction coefficient in 1/m
 - pixel_pitch and focal_length in meters
"""

from typing import Tuple
import numpy as np
from dataclass.opticDataClass import OpticSensorsParameters
from model.units import assert_positive, db2lin
from dataclass.opticDataClass import OpticSensorsParameters


class OpticalSensorModel:
    def __init__(self, params: OpticSensorsParameters):
        self.params = params

    def effective_range(self) -> float:
        I_0 = self.params.I_0
        I_min = self.params.I_min
        beta = self.params.beta
        if I_min <= 0 or I_0 <= 0:
            raise ValueError("I_0 and I_min must be > 0 (W/m^2)")
        if beta <= 0:
            return np.inf  # no attenuation

        R_eff = -np.log(I_min / I_0) / beta
        return R_eff

    def angle_error_imaging(self) -> Tuple[float, float, float]:
        lambda_val = self.params.wavelength
        D = self.params.D
        sigma_diff_theta = 1.22 * lambda_val / D

        d = self.params.pixel_pitch
        f = self.params.focal_length
        SNR = db2lin(self.params.SNR_db)
        sigma_smpl_theta = (d / f) * (1 / np.sqrt(SNR))

        sigma_tot_theta = np.sqrt(sigma_diff_theta ** 2 + sigma_smpl_theta ** 2)

        return sigma_diff_theta, sigma_smpl_theta, sigma_tot_theta

    def angle_error_non_imaging_psd(self) -> float:
        f = self.params.focal_length
        spot_diameter = self.params.spot_diameter
        SNR = db2lin(self.params.SNR_db)
        sigma_pos_X = self.params.sigma_pos_x

        sigma_noise_X = spot_diameter / np.sqrt(SNR)

        sigma_tot_X = np.sqrt(sigma_noise_X ** 2 + sigma_pos_X ** 2)

        sigma_theta = sigma_tot_X / f

        return sigma_theta

    def angle_error(self) -> float:

        if self.params.sensor_type == "imaging":
            _, _, sigma_theta = self.angle_error_imaging()
            return sigma_theta
        elif self.params.sensor_type == "non-imaging":
            return self.angle_error_non_imaging_psd()
        else:
            raise ValueError(f"Unknown sensor type: {self.params.sensor_type}")
