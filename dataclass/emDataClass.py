from dataclasses import dataclass

from model.units import assert_positive, freq_to_wavelength, db2lin


@dataclass
class EmSensorsParameters:
    P_t: float
    G_t: float
    G_r: float
    frequency: float
    P_r_min: float
    L_system: float
    alpha: float
    D: float  # aperture size
    SNR_db: float

    # for passive EM sensor
    cross_section: float

    sensor_type: str = "Active"  # it could be passive too, but we just only consider active sensor here

    def __post_init__(self):
        assert_positive("P_t", self.P_t)
        assert_positive("G_t", self.G_t)
        assert_positive("G_r", self.G_r)
        assert_positive("frequency", self.frequency)
        assert_positive("P_r_min", self.P_r_min)
        assert_positive("L_system", self.L_system)
        if self.alpha < 0:
            raise ValueError("alpha (attenuation coefficient) should be >= 0 (units: 1/m)")
        assert_positive("D (aperture diameter)", self.D)
        assert_positive("cross_section", self.cross_section)

    @property
    def wavelength(self) -> float:
        return freq_to_wavelength(self.frequency)

    @property
    def snr_linear(self) -> float:
        return db2lin(self.SNR_db)
