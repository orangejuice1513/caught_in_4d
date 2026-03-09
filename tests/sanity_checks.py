from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from sim.env import DroneEnv, DroneConfig, SimConfig
from sim.controller import GeometricController, ReferenceState


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "assets" / "quad.urdf"


def _repo_urdf() -> str:
    if not DEFAULT_URDF.exists():
        pytest.skip(f"Missing URDF at {DEFAULT_URDF}")
    return str(DEFAULT_URDF)


def make_env(*, z0: float = 0.8, gui: bool = False) -> DroneEnv:
    return DroneEnv(
        sim_cfg=SimConfig(
            dt=1 / 240,
            physics_substeps=2,
            gui=gui,
            enable_ground=True,
        ),
        drone_cfg=DroneConfig(
            urdf_path=_repo_urdf(),
            expected_mass_kg=1.35,
            start_pos_w=np.array([0.0, 0.0, z0], dtype=float),
        ),
    )


def step_many(env: DroneEnv, rpm_cmd: np.ndarray, steps: int):
    truth = env.get_truth_state()
    for _ in range(steps):
        truth = env.step(rpm_cmd)
    return truth


def test_reset_returns_finite_state():
    env = make_env(z0=0.8)
    try:
        truth = env.reset()
        assert truth.step_idx == 0
        assert truth.t == pytest.approx(0.0)
        assert np.all(np.isfinite(truth.pos_w))
        assert np.all(np.isfinite(truth.vel_w))
        assert np.all(np.isfinite(truth.quat_wb_xyzw))
        assert np.linalg.norm(truth.quat_wb_xyzw) == pytest.approx(1.0, abs=1e-6)
        assert truth.pos_w[2] == pytest.approx(0.8, abs=1e-6)
        assert env.get_mass_kg() > 0.0
        assert np.all(env.get_inertia_diag() > 0.0)
    finally:
        env.close()


def test_zero_rpm_falls():
    env = make_env(z0=1.5)
    try:
        truth0 = env.reset()
        truth1 = step_many(env, np.zeros(4), steps=120)  # 0.5 s
        assert truth1.pos_w[2] < truth0.pos_w[2] - 0.1
        assert truth1.vel_w[2] < -0.5
    finally:
        env.close()


def test_set_motor_rpm_command_clips_bounds():
    env = make_env(z0=0.5)
    try:
        env.reset()
        env.set_motor_rpm_command(np.array([-1000.0, 1e9, 100.0, 200.0]))
        truth = env.step()
        min_rpm = env.drone_cfg.motor.min_rpm
        max_rpm = env.drone_cfg.motor.max_rpm
        assert np.all(truth.motor_rpm_cmd >= min_rpm)
        assert np.all(truth.motor_rpm_cmd <= max_rpm)
        assert truth.motor_rpm_cmd[0] == pytest.approx(min_rpm)
        assert truth.motor_rpm_cmd[1] == pytest.approx(max_rpm)
    finally:
        env.close()


def test_symmetric_rpm_does_not_create_large_lateral_drift_or_yaw_spin():
    env = make_env(z0=1.5)
    try:
        env.reset()
        hover_rpm = env.estimate_hover_rpm() * 1.03
        cmd = np.full(4, hover_rpm, dtype=float)
        truth = step_many(env, cmd, steps=240)  # 1 s

        # Equal rotor commands should mostly produce vertical behavior.
        xy_disp = np.linalg.norm(truth.pos_w[:2])
        yaw_rate_mag = abs(truth.omega_b[2])
        roll_pitch_mag = np.linalg.norm(truth.euler_rpy_wb[:2])

        assert xy_disp < 0.2
        assert yaw_rate_mag < 1.0
        assert roll_pitch_mag < 0.5
    finally:
        env.close()


def test_controller_climbs_toward_hover_target():
    env = make_env(z0=0.3)
    try:
        truth = env.reset()
        ctrl = GeometricController.from_env(env)
        ref = ReferenceState(
            pos_w=np.array([0.0, 0.0, 1.5]),
            vel_w=np.zeros(3),
            acc_w=np.zeros(3),
            yaw=0.0,
            yaw_rate=0.0,
        )

        z0 = truth.pos_w[2]
        for _ in range(720):  # 3 s
            truth = env.get_truth_state()
            rpm_cmd = ctrl.compute_rpm(truth, ref, dt=env.sim_cfg.dt)
            truth = env.step(rpm_cmd)

        assert truth.pos_w[2] > z0 + 0.4
        assert truth.pos_w[2] > 0.9
    finally:
        env.close()


def test_controller_recovers_from_position_disturbance():
    env = make_env(z0=1.5)
    try:
        env.reset()
        ctrl = GeometricController.from_env(env)
        ref = ReferenceState(
            pos_w=np.array([0.0, 0.0, 1.5]),
            vel_w=np.zeros(3),
            acc_w=np.zeros(3),
            yaw=0.0,
            yaw_rate=0.0,
        )

        # Let it settle near the target first.
        for _ in range(720):
            truth = env.get_truth_state()
            truth = env.step(ctrl.compute_rpm(truth, ref, dt=env.sim_cfg.dt))

        # Inject a hard state disturbance.
        disturbed = env.set_state(
            pos_w=np.array([0.8, -0.6, 1.0]),
            quat_wb_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            vel_w=np.array([1.0, -0.5, 0.2]),
            omega_w=np.zeros(3),
        )
        err0 = np.linalg.norm(disturbed.pos_w - ref.pos_w)

        for _ in range(960):  # 4 s
            truth = env.get_truth_state()
            truth = env.step(ctrl.compute_rpm(truth, ref, dt=env.sim_cfg.dt))

        err1 = np.linalg.norm(truth.pos_w - ref.pos_w)
        assert err1 < 0.4 * err0
        assert err1 < 0.5
    finally:
        env.close()
