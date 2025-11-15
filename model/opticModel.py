import warnings
from typing import Tuple
import numpy as np

from dataclass.opticDataClass import OpticSensorsParameters


class OpticalSensorModel:
    def __init__(self, params: OpticSensorsParameters):
        self.params = params

    def effectiveRange(self) -> float:
        I_0 = self.params.I_0
        I_min = self.params.I_min
        beta = self.params.beta

        if I_0 <= I_min:
            warnings.warn("Initial intensity is less than minimum detectable intensity")
            return 0.0

        R_eff = (1 / beta) * np.log(I_0 / I_min)
        return R_eff

    def anglErrorImaging(self) -> Tuple[float, float, float]:
        lambda_val = self.params.wavelength
        D = self.params.D
        sigma_diff_theta = 1.22 * lambda_val / D

        d = self.params.pixel_pitch
        f = self.params.focal_length
        SNR = self.params.SNR_linear
        sigma_smpl_theta = (d / f) * (1 / np.sqrt(SNR))

        sigma_tot_theta = np.sqrt(sigma_diff_theta ** 2 + sigma_smpl_theta ** 2)

        return sigma_diff_theta, sigma_smpl_theta, sigma_tot_theta

    def angleErrorNonImagingPsd(self) -> float:
        f = self.params.focal_length
        spot_diameter = self.params.spot_diameter
        SNR = self.params.SNR_linear
        sigma_pos_X = self.params.sigma_pos_X

        sigma_noise_X = spot_diameter / np.sqrt(SNR)

        sigma_tot_X = np.sqrt(sigma_noise_X ** 2 + sigma_pos_X ** 2)

        sigma_theta = sigma_tot_X / f

        return sigma_theta

    def angleError(self) -> float:

        if self.params.sensor_type == "imaging":
            _, _, sigma_theta = self.anglErrorImaging()
            return sigma_theta
        elif self.params.sensor_type == "non-imaging":
            return self.angleErrorNonImagingPsd()
        else:
            raise ValueError(f"Unknown sensor type: {self.params.sensor_type}")
