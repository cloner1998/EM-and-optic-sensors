"""
TerrainProfile:
 - x_positions: 1D numpy array of horizontal distances (meters) from transmitter toward receiver;
   must be non-negative and strictly increasing.
 - h_terrain: 1D numpy array of terrain heights above reference (meters), same length as x_positions
 - h_transmitter, h_receiver: heights (meters)
"""

from dataclasses import dataclass
import numpy as np
from model.units import assert_positive

@dataclass
class TerrainProfile:
    x_positions: np.ndarray
    h_terrain: np.ndarray
    h_transmitter: float
    h_receiver: float

    def __post_init__(self):
        if self.x_positions.ndim != 1:
            raise ValueError("x_positions must be a 1D numpy array (meters).")
        if self.h_terrain.ndim != 1:
            raise ValueError("h_terrain must be a 1D numpy array (meters).")
        if len(self.x_positions) != len(self.h_terrain):
            raise ValueError("x_positions and h_terrain must have the same length.")
        if not np.all(np.diff(self.x_positions) >= 0):
            raise ValueError("x_positions must be non-decreasing (meters).")
        assert_positive("h_transmitter", self.h_transmitter)
        assert_positive("h_receiver", self.h_receiver)
