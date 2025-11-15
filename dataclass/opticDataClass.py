@property
class OpticSensorsParameters:
    I_0: float
    I_min: float
    wavelength: float
    beta: float
    sensor_type: str = "imaging" #it could be non-imaging

    #imgaing sensor parameter
    D: float
    pixel_pitch: float = 5e-6
    focal_length: float = 0.05
    SNR_db: float = 30.0

    #non-imaging sensors
    #ToDo(we should investigate more on this parameter for non-imaging sensors)
    spot_diameter: float = 1e-3
    sigma_pos_x: float = 1e-4

    @property
    def SNRLinear(self) -> float:
        return 10**(self.SNR_db / 10)

