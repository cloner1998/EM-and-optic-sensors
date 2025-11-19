"""
EM sensor models: FSPL, passive received power, active radar eq., and angle error.

All distances in meters, frequency in Hz, gain linear, powers in Watts, SNR_db in dB.
"""

from dataclass.emDataClass import EmSensorsParameters
import numpy as np
from model.units import assert_positive, lin2db


class EMSensorsModel:
    def __init__(self, params: EmSensorsParameters):
        self.params = params

    def free_space_path_loss_linear(self, R: float) -> float:
        assert_positive("R", R)
        lam = self.params.wavelength
        return (4.0 * np.pi * R / lam) ** 2

    def free_space_path_loss_db(self, R: float) -> float:
        linear = self.free_space_path_loss_linear(R)
        return lin2db(linear)

    def received_power_passive(self, R: float) -> float:
        """
            Passive received power (e.g., isotropic scatterer scenario) simplified:
            Pr = Pt * Gt * Gr / (FSPL_linear * L_system) * exp(-alpha * R)
            - alpha is linear attenuation coefficient (1/m)
        """
        assert_positive("R", R)
        P_t = self.params.P_t
        G_t = self.params.G_t
        G_r = self.params.G_r
        wavelength = self.params.wavelength
        Lsys = self.params.L_system
        alpha = self.params.alpha

        fspl = self.free_space_path_loss_linear(R)
        path_atten = np.exp(-alpha * R)

        P_r = (P_t * G_t * G_r / (fspl * Lsys)) * path_atten

        return P_r

    def received_power_active_monostatic(self, R: float):
        """
            Monostatic radar equation (linear):
            Pr = Pt * Gt * Gr * lambda^2 * sigma / ((4*pi)^3 * R^4 * L_system) * exp(-2*alpha*R)
            where sigma is radar cross section (m^2)
        """

        assert_positive("R", R)
        P_t = self.params.P_t
        G_t = self.params.G_t
        G_r = self.params.G_r
        lam = self.params.wavelength
        sigma = self.params.cross_section
        Lsys = self.params.L_system
        alpha = self.params.alpha

        denom = (4.0 * np.pi) ** 3 * (R ** 4)
        P_r = P_t * G_t * G_r * (lam ** 2) * sigma / (denom * Lsys) * np.exp(-2.0 * alpha * R)
        # P_r = (P_t * (G ** 2) * cross_section * (wavelength ** 2)) / ((np.pi ** 3) * R ** 4)
        # above formula does not consider attenuation and system loss so i just deleted
        return P_r

    def effective_range(self) -> float:
        if self.params.sensor_type.lower() == "active":
            R_min = self.bisection(self.received_power_active_monostatic)
        elif self.params.sensor_type.lower() == "passive":
            R_min = self.bisection(self.received_power_passive)
        else:
            raise NotImplementedError
        return R_min

    def bisection(self, func, R_min=1.0, R_max=1e6, tolerance=1.0) -> float:
        while R_max - R_min > tolerance:
            R_mid = (R_max + R_min) / 2
            P_r = func(R_mid)

            if P_r > self.params.P_r_min:
                R_min = R_mid
            else:
                R_max = R_mid
        return R_min

    def angle_error(self) -> float:
        """
            Approximate angle RMS error (radians) for a single-aperture phased sensor:
            sigma_theta ≈ lambda / (2*pi*D*sqrt(SNR_linear))
            Returns radians.
        """
        lam = self.params.wavelength
        D = self.params.D
        SNR = self.params.snr_linear
        assert_positive("SNR", SNR)
        sigma_theta = lam / (2.0 * np.pi * D * np.sqrt(SNR))
        return sigma_theta
