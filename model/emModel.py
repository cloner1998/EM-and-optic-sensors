
from dataclass.emDataClass import EmSensorsParameters
import numpy as np


class EMSensorsModel:
    def __init__(self, params: EmSensorsParameters):
        self.params = params

    def free_space_path_loss(self, R: float):
        return np.power(4 * np.pi * R / self.params.wavelength, 2)

    def received_power_passive(self, R: float):
        P_t = self.params.P_t
        G_t = self.params.G_t
        G_r = self.params.G_r
        wavelength = self.params.wavelength
        L = self.params.L_system
        alpha = self.params.alpha

        P_r = (P_t * G_t * G_r * np.power((wavelength / (4 * np.pi * R)), 2) *
               np.exp(-alpha * R) / L)

        return P_r


    def received_power_active(self, R: float):
        P_t = self.params.P_t
        G = self.params.G_t
        wavelength = self.params.wavelength
        cross_section = self.params.cross_section

        P_r = (P_t * (G**2) * cross_section * (wavelength ** 2))/((np.pi ** 3) * R ** 4)
        return P_r


    def effective_range(self) -> float:
        R_min, R_max = 1.0, 1e6
        tolerance = 1.0
        if self.params.sensor_type.lower() == "active":
            while R_max - R_min > tolerance:
                R_mid = (R_max + R_min) / 2
                P_r = self.received_power_active(R_mid)

                if P_r > self.params.P_r_min:
                    R_min = R_mid
                else:
                    R_max = R_mid
        elif self.params.sensor_type.lower() == "passive":
            while R_max - R_min > tolerance:
                R_mid = (R_max + R_min) / 2
                P_r = self.received_power_passive(R_mid)

                if P_r > self.params.P_r_min:
                    R_min = R_mid
                else:
                    R_max = R_mid
        else:
            raise NotImplementedError
        return R_min

    def angle_error(self) -> float:
        lambda_val = self.params.wavelength
        D = self.params.D
        SNR = self.params.snr_linear()

        sigma_theta = lambda_val / (2 * np.pi * D * np.sqrt(SNR))
        return sigma_theta

