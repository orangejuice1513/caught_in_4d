from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from src.filters.hover_ukf import HoverUKF
from src.filters.race_ukf import RaceUKF
from src.filters.adaptive_ukf import AdaptiveUKF, AdaptiveUKFConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay one sim log through Hover/Race/Adaptive UKFs.")
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to sim_run.npz. Default: <repo>/logs/sim_run.npz",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. Default: <repo>/logs/filter_results.npz",
    )
    return p


def default_log_path(repo_root: Path) -> Path:
    return repo_root / "logs" / "sim_run.npz"


def default_out_path(repo_root: Path) -> Path:
    return repo_root / "logs" / "filter_results.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def make_initial_state_from_truth(truth_pos: np.ndarray, truth_vel: np.ndarray, truth_quat: np.ndarray) -> np.ndarray:
    # Expected order from earlier ukf_core sketch:
    # [px, py, pz, vx, vy, vz, qx, qy, qz, qw]
    q = np.asarray(truth_quat, dtype=float).reshape(4)
    if np.allclose(q, 0.0):
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    # Your env log stores quat_wb. Earlier helpers used [qx, qy, qz, qw] in state.
    # If your truth_quat is [qw, qx, qy, qz], reorder here.
    if abs(q[0]) <= 1.0 and abs(q[3]) <= 1.0:
        # Heuristic: env.py earlier used quat_wb = [w, x, y, z]
        qx, qy, qz, qw = q[1], q[2], q[3], q[0]
    else:
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    return np.array(
        [
            truth_pos[0], truth_pos[1], truth_pos[2],
            truth_vel[0], truth_vel[1], truth_vel[2],
            qx, qy, qz, qw,
        ],
        dtype=float,
    )


def make_initial_covariance() -> np.ndarray:
    return np.diag(
        [
            0.20, 0.20, 0.20,   # position
            0.50, 0.50, 0.50,   # velocity
            0.05, 0.05, 0.05,   # attitude error-state scale proxy
        ]
    )


def pos_from_state(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[:3], dtype=float)


def compute_rmse(est: np.ndarray, truth: np.ndarray) -> float:
    err = est - truth
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    log_path = Path(args.log).expanduser().resolve() if args.log else default_log_path(repo_root)
    out_path = Path(args.out).expanduser().resolve() if args.out else default_out_path(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    D = load_npz(log_path)

    t = D["t"]
    truth_pos = D["truth_pos_w"]
    truth_vel = D["truth_vel_w"]
    truth_quat = D["truth_quat_wb"]

    imu_valid = D["imu_valid"].astype(bool)
    imu_acc = D["imu_accel_mps2"]
    imu_gyro = D["imu_gyro_radps"]

    vio_valid = D["vio_valid"].astype(bool)
    vio_pos = D["vio_pos_w_m"]

    telemetry_rpm_sq_sum = D["telemetry_rpm_sq_sum"]
    telemetry_specific_force_mag = D["telemetry_specific_force_mag_mps2"]

    x0 = make_initial_state_from_truth(
        truth_pos=truth_pos[0],
        truth_vel=truth_vel[0],
        truth_quat=truth_quat[0],
    )
    P0 = make_initial_covariance()

    hover = HoverUKF()
    race = RaceUKF()
    adaptive = AdaptiveUKF(
        AdaptiveUKFConfig(
            motor_max_rpm=12000.0,
        )
    )

    hover.initialize(x0, P0)
    race.initialize(x0, P0)
    adaptive.initialize(x0, P0)

    n = len(t)

    hover_x = np.full((n, len(x0)), np.nan, dtype=float)
    race_x = np.full((n, len(x0)), np.nan, dtype=float)
    adaptive_x = np.full((n, len(x0)), np.nan, dtype=float)

    hover_p = np.full((n, P0.shape[0], P0.shape[1]), np.nan, dtype=float)
    race_p = np.full((n, P0.shape[0], P0.shape[1]), np.nan, dtype=float)
    adaptive_p = np.full((n, P0.shape[0], P0.shape[1]), np.nan, dtype=float)

    adaptive_R = np.full(n, np.nan, dtype=float)

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if dt <= 0.0:
            continue

        if imu_valid[k]:
            hover.predict(imu_acc[k], imu_gyro[k], dt)
            race.predict(imu_acc[k], imu_gyro[k], dt)
            adaptive.predict(imu_acc[k], imu_gyro[k], dt)

        if vio_valid[k]:
            hover.update_vio(vio_pos[k])
            race.update_vio(vio_pos[k])
            adaptive.update_vio(
                vio_pos_w_m=vio_pos[k],
                rpm_sq_sum=float(telemetry_rpm_sq_sum[k]),
                specific_force_mag_mps2=float(telemetry_specific_force_mag[k]),
            )

        hover_x[k] = hover.state()
        race_x[k] = race.state()
        adaptive_x[k] = adaptive.state()

        hover_p[k] = hover.covariance()
        race_p[k] = race.covariance()
        adaptive_p[k] = adaptive.covariance()

        adaptive_R[k] = adaptive.last_R_scalar

    hover_pos = hover_x[:, :3]
    race_pos = race_x[:, :3]
    adaptive_pos = adaptive_x[:, :3]

    valid_hover = np.all(np.isfinite(hover_pos), axis=1)
    valid_race = np.all(np.isfinite(race_pos), axis=1)
    valid_adapt = np.all(np.isfinite(adaptive_pos), axis=1)

    results = {
        "t": t,
        "truth_pos_w": truth_pos,
        "hover_state": hover_x,
        "race_state": race_x,
        "adaptive_state": adaptive_x,
        "hover_cov": hover_p,
        "race_cov": race_p,
        "adaptive_cov": adaptive_p,
        "adaptive_R_scalar": adaptive_R,
        "hover_pos_w": hover_pos,
        "race_pos_w": race_pos,
        "adaptive_pos_w": adaptive_pos,
        "hover_valid": valid_hover.astype(np.int32),
        "race_valid": valid_race.astype(np.int32),
        "adaptive_valid": valid_adapt.astype(np.int32),
    }

    np.savez_compressed(out_path, **results)

    summary = {
        "hover_rmse_m": compute_rmse(hover_pos[valid_hover], truth_pos[valid_hover]) if np.any(valid_hover) else None,
        "race_rmse_m": compute_rmse(race_pos[valid_race], truth_pos[valid_race]) if np.any(valid_race) else None,
        "adaptive_rmse_m": compute_rmse(adaptive_pos[valid_adapt], truth_pos[valid_adapt]) if np.any(valid_adapt) else None,
    }

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved filter replay results to: {out_path}")
    print(f"Saved summary to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()