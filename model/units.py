# model/units.py
"""
Small unit helpers and validators used across the repo.

Conventions:
 - distances/lengths: meters (m)
 - frequency: Hertz (Hz)
 - wavelength: meters (m)
 - power: Watts (W)
 - gains: linear (not dB) unless variable name ends with "_db"
 - SNR_db: decibels
 - angles: radians (unless documented)
"""

import math

c = 299792458.0  # speed of light (m/s)

def freq_to_wavelength(f_hz: float) -> float:
    if f_hz <= 0:
        raise ValueError("frequency must be > 0 Hz")
    return c / f_hz

def wavelength_to_freq(lambda_m: float) -> float:
    if lambda_m <= 0:
        raise ValueError("wavelength must be > 0 m")
    return c / lambda_m

def db2lin(db: float) -> float:
    return 10.0 ** (db / 10.0)

def lin2db(x: float) -> float:
    if x <= 0:
        raise ValueError("lin2db: input must be > 0")
    return 10.0 * math.log10(x)

def assert_positive(name: str, value: float) -> None:
    if value is None:
        raise ValueError(f"{name} is None")
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")
