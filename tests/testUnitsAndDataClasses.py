import pytest


def test_optic_params_bad_wavelength():
    from dataclass.opticDataClass import OpticSensorsParameters
    with pytest.raises(ValueError):
        OpticSensorsParameters(I_0=1.0, I_min=1e-6, wavelength=0.0, beta=0.1)

    from dataclass.emDataClass import EmSensorsParameters
    with pytest.raises(ValueError):
        EmSensorsParameters(P_t=0, G_t=1, G_r=1, frequency=1e9, P_r_min=1e-12, L_system=1, alpha=0, D=0.1, SNR_db=0,
                            cross_section=1)
