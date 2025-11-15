from dataclasses import dataclass
import numpy as np

@dataclass
class TerrainProfile:
    x_positions: np.ndarray
    h_terrain: np.ndarray
    h_transmitter: float
    h_receiver: float
