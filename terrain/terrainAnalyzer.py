from typing import Tuple

from dataclass.terrain import TerrainProfile
import numpy as np


class TerrainAnalyzer:
    def __init__(self, terrain: TerrainProfile):
        self.terrain = terrain

    def line_of_sight_height(self, x: np.ndarray, R: float) -> np.ndarray:
        h_t = self.terrain.h_transmitter
        h_r = self.terrain.h_receiver
        h_LOS = h_t + (h_r - h_t) * (x / R)
        return h_LOS

    def check_los_clearance(self, R: float) -> Tuple[bool, float]:
        x = self.terrain.x_positions[self.terrain.x_positions <= R]
        h_terrain = self.terrain.h_terrain[:len(x)]
        h_LOS = self.line_of_sight_height(x, R)

        clearance = h_LOS - h_terrain
        min_clearance = np.min(clearance)
        is_clear = min_clearance > 0

        return is_clear, min_clearance

    def find_max_clear_range(self, R_max: float, num_points: int = 1000) -> float:
        R_test = np.linspace(1, R_max, num_points)

        for R in R_test:
            is_clear, _ = self.check_los_clearance(R)
            if not is_clear:
                return R

        return R_max

    def minimum_elevation_angle(self) -> float:
        x = self.terrain.x_positions[self.terrain.x_positions > 0]
        h_terrain = self.terrain.h_terrain[1:len(x) + 1]
        h_rec = self.terrain.h_receiver

        theta_min = (h_terrain - h_rec) / x
        theta_LOS_min = np.max(theta_min)

        return theta_LOS_min

    def fresnel_radius(d1: float, d2: float, wavelength: float) -> float:
        r_F = np.sqrt(wavelength * d1 * d2 / (d1 + d2))
        return r_F
