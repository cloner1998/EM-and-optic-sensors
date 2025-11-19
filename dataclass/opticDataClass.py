from dataclasses import dataclass
from model.units import assert_positive, db2lin

@dataclass
class OpticSensorsParameters:
    I_0: float              # W/m^2, source irradiance
    I_min: float            # W/m^2, minimum detectable irradiance
    wavelength: float       # m
    beta: float             # extinction coefficient (1/m)
    D: float = 0.1          # aperture diameter (m)
    pixel_pitch: float = 5e-6
    focal_length: float = 0.05
    SNR_db: float = 30.0
    spot_diameter: float = 1e-3
    sigma_pos_x: float = 1e-4
    sensor_type: str = "imaging"

    def __post_init__(self):
        assert_positive("I_0", self.I_0)
        assert_positive("I_min", self.I_min)
        assert_positive("wavelength", self.wavelength)
        if self.beta < 0:
            raise ValueError("beta (extinction coefficient) should be >= 0 (1/m)")
        assert_positive("D (aperture)", self.D)
        assert_positive("pixel_pitch", self.pixel_pitch)
        assert_positive("focal_length", self.focal_length)
