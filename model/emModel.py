from dataclass.emDataClass import EmSensorsParameters
import numpy as np


class EMSensorsModel:
    def __init__(self, params: EmSensorsParameters):
        self.params = params

    def freeSpacePathLoss(self, R: float):
        return np.power(4 * np.pi * R / self.params.wavelength, 2)

    def receivedPower(self, R: float):
        P_t = self.params.P_t
        G_t = self.params.G_t
        G_r = self.params.G_r
        wavelength = self.params.wavelength
        L = self.params.L_system
        alpha = self.params.alpha

        P_r = (np.power(P_t * G_t * G_r * (wavelength / (4 * np.pi * R)), 2) *
               np.exp(-alpha * R) / L)

        return P_r

    # ToDo(implement effective range for passive EM sensors)

    def effectiveRange(self) -> float:
        R_min, R_max = 1.0, 1e6
        tolerance = 1.0

        while R_max - R_min > tolerance:
            R_mid = (R_max + R_min) / 2
            P_r = self.receivedPower(R_mid)

            if P_r > self.params.P_r_min:
                R_min = R_mid
            else:
                R_max = R_mid

        return R_min

    def angleError(self) -> float:
        """Calculate RMS angular error (radians) using Equation 10"""
        lambda_val = self.params.wavelength
        D = self.params.D
        SNR = self.params.SNRLinear()

        sigma_theta = lambda_val / (2 * np.pi * D * np.sqrt(SNR))
        return sigma_theta

