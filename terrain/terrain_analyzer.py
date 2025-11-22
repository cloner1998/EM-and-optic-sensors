from typing import Tuple

from dataclass.terrain import TerrainProfile
import numpy as np

from model.units import assert_positive


class TerrainAnalyzer:
    def __init__(self, terrain: TerrainProfile):
        self.terrain = terrain

    def line_of_sight(self, x: np.ndarray, R: float) -> np.ndarray:
        assert_positive("R",R)
        h_t = self.terrain.h_transmitter
        h_r = self.terrain.h_receiver
        h_LOS = h_t + (h_r - h_t) * (x / R)
        return h_LOS

    def check_los_clearance(self, R: float) -> Tuple[bool, float]:
        x = self.terrain.x_positions[self.terrain.x_positions <= R]
        h_terrain = self.terrain.h_terrain[:len(x)]
        h_LOS = self.line_of_sight(x, R)

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
        """
            Return minimum elevation angle (radians) from receiver to clear all terrain profile points.
            Uses the terrain sample points in self.terrain.x_positions/h_terrain.
        """
        x = self.terrain.x_positions
        h_terrain = self.terrain.h_terrain
        # horizontal distance from receiver: assume x runs from 0..R; if transmitter at x=0, remove it.
        # Compute required angle such that line from receiver clears each terrain point:
        # angle_i = arctan((h_terrain[i] - h_receiver) / (R - x[i]))  if x[i] is distance from transmitter.
        R_total = x[-1]
        h_rec = self.terrain.h_receiver
        # distances from receiver to each terrain sample:
        d_from_rec = R_total - x
        # avoid division by zero for transmitter/receiver points:
        # we should avoid zero because it is location of the receiver itself
        mask = d_from_rec > 0
        if not np.any(mask):
            return 0.0
        angles = np.arctan((h_terrain[mask] - h_rec) / d_from_rec[mask])
        theta_min = np.max(angles)
        return theta_min

    def fresnel_radius(d1: float, d2: float, wavelength: float) -> float:
        r_F = np.sqrt(wavelength * d1 * d2 / (d1 + d2))
        return r_F
