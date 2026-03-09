from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.filters.ukf_core import (
    UnscentedKalmanFilter,
    DroneStateSpaceModel,
    make_process_noise,
    make_position_measurement_noise,
)


@dataclass(slots=True)
class RaceUKFConfig:
    # Higher fixed-noise tuning for aggressive flight
    pos_process_var: float = 5e-5
    vel_process_var: float = 5e-2
    att_process_var: float = 5e-5
    vio_pos_var: float = 0.10 ** 2


class RaceUKF:
    def __init__(self, cfg: RaceUKFConfig | None = None) -> None:
        self.cfg = cfg or RaceUKFConfig()
        self.model = DroneStateSpaceModel()
        self.ukf = UnscentedKalmanFilter(self.model)

        self.Q = make_process_noise(
            pos_var=self.cfg.pos_process_var,
            vel_var=self.cfg.vel_process_var,
            att_var=self.cfg.att_process_var,
        )
        self.R = make_position_measurement_noise(self.cfg.vio_pos_var)

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.ukf.initialize(x0, P0)

    def predict(self, imu_accel_mps2: np.ndarray, imu_gyro_radps: np.ndarray, dt: float) -> None:
        u = np.concatenate([imu_accel_mps2, imu_gyro_radps]).astype(float)
        self.ukf.predict(u=u, dt=dt, Q=self.Q)

    def update_vio(self, vio_pos_w_m: np.ndarray) -> None:
        self.ukf.update(z=np.asarray(vio_pos_w_m, dtype=float), R=self.R)

    def state(self) -> np.ndarray:
        return self.ukf.state.mean.copy()

    def covariance(self) -> np.ndarray:
        return self.ukf.state.cov.copy()