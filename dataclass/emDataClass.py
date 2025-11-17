from dataclasses import dataclass


@dataclass
class EmSensorsParameters:
    P_t: float
    G_t: float
    G_r: float
    frequency: float
    P_r_min: float
    L_system: float
    alpha: float
    D: float
    SNR_db: float

    # for passive EM sensor
    cross_section: float

    sensor_type: str = "Active"  # it could be passive too, but we just only consider active sensor here

    @property
    def wavelength(self) -> float:
        c = 3e8
        return c / self.frequency

    def snr_linear(self) -> float:
        return 10 ** (self.SNR_db / 10)
