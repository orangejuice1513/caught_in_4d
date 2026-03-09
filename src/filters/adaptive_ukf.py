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
class AdaptiveUKFConfig:
    # Base process noise
    pos_process_var: float = 3e-5
    vel_process_var: float = 2e-2
    att_process_var: float = 3e-5

    # Base measurement noise floor
    vio_pos_var: float = 0.05 ** 2

    # Paper-inspired adaptive scaling:
    # R_k = R_floor + alpha * sum(RPM_i^2) + beta * ||a||
    #
    # We normalize sum(RPM^2) by 4 * max_rpm^2 so alpha stays sane.
    alpha_rpm_var: float = 0.50
    beta_g_var: float = 0.03

    motor_max_rpm: float = 12000.0
    max_vio_var: float = 2.0

    def rpm_sq_norm(self, rpm_sq_sum: float) -> float:
        return float(rpm_sq_sum / (4.0 * self.motor_max_rpm ** 2))


class AdaptiveUKF:
    def __init__(self, cfg: AdaptiveUKFConfig | None = None) -> None:
        self.cfg = cfg or AdaptiveUKFConfig()
        self.model = DroneStateSpaceModel()
        self.ukf = UnscentedKalmanFilter(self.model)

        self.Q = make_process_noise(
            pos_var=self.cfg.pos_process_var,
            vel_var=self.cfg.vel_process_var,
            att_var=self.cfg.att_process_var,
        )

        self.last_R_scalar: float = self.cfg.vio_pos_var

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.ukf.initialize(x0, P0)

    def predict(self, imu_accel_mps2: np.ndarray, imu_gyro_radps: np.ndarray, dt: float) -> None:
        u = np.concatenate([imu_accel_mps2, imu_gyro_radps]).astype(float)
        self.ukf.predict(u=u, dt=dt, Q=self.Q)

    def compute_R(
        self,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> np.ndarray:
        rpm_term = self.cfg.alpha_rpm_var * self.cfg.rpm_sq_norm(rpm_sq_sum)
        g_term = self.cfg.beta_g_var * float(specific_force_mag_mps2)

        R_scalar = self.cfg.vio_pos_var + rpm_term + g_term
        R_scalar = float(np.clip(R_scalar, self.cfg.vio_pos_var, self.cfg.max_vio_var))
        self.last_R_scalar = R_scalar
        return make_position_measurement_noise(R_scalar)

    def update_vio(
        self,
        vio_pos_w_m: np.ndarray,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> None:
        R = self.compute_R(
            rpm_sq_sum=rpm_sq_sum,
            specific_force_mag_mps2=specific_force_mag_mps2,
        )
        self.ukf.update(z=np.asarray(vio_pos_w_m, dtype=float), R=R)

    def state(self) -> np.ndarray:
        return self.ukf.state.mean.copy()

    def covariance(self) -> np.ndarray:
        return self.ukf.state.cov.copy()