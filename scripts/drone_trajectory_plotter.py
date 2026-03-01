#!/usr/bin/env python3
"""
3D drone trajectory plotter with orientation visualization.
Reads from flight-01a-ellipse_cam_ts_sync.csv and shows a scrollable timeline
with a simple drone figure at the current frame.
Run from project root: python scripts/drone_trajectory_plotter.py
Optional: python scripts/drone_trajectory_plotter.py path/to/other_cam_ts_sync.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Resolve paths: script in scripts/, data in data/autonomous/flight-01a-ellipse/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / "data/autonomous/flight-01a-ellipse/flight-01a-ellipse_cam_ts_sync.csv"


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Build 3x3 rotation matrix from Euler angles (roll, pitch, yaw) in radians.
    Convention: intrinsic rotations Z (yaw) -> Y (pitch) -> X (roll).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def get_drone_vertices(scale=0.08):
    """
    Simple quadcopter in body frame: center + 4 arms (X configuration).
    Body frame: X forward, Y left, Z up. Returns list of line segments
    as ((x0,x1), (y0,y1), (z0,z1)).
    """
    arm_len = scale
    center = (0, 0, 0)
    # Four arm endpoints (X layout when viewed from above)
    arm_pts = [
        (arm_len, -arm_len, 0),   # front-right
        (arm_len, arm_len, 0),    # front-left
        (-arm_len, arm_len, 0),   # back-left
        (-arm_len, -arm_len, 0),  # back-right
    ]
    lines = []
    for p in arm_pts:
        lines.append(([center[0], p[0]], [center[1], p[1]], [center[2], p[2]]))
    # Nose indicator (short line along body +X)
    nose_len = scale * 0.6
    lines.append(([0, nose_len], [0, 0], [0, 0]))
    return lines


def transform_drone_lines(lines, R, tx, ty, tz):
    """Rotate and translate body-frame lines by R and (tx, ty, tz)."""
    out = []
    for (xs, ys, zs) in lines:
        pts = np.column_stack([xs, ys, zs])
        rotated = (R @ pts.T).T
        out.append((
            rotated[:, 0] + tx,
            rotated[:, 1] + ty,
            rotated[:, 2] + tz,
        ))
    return out


def load_data(csv_path):
    """Load CSV and extract position and orientation columns."""
    df = pd.read_csv(csv_path)
    required = ["drone_x", "drone_y", "drone_z", "drone_roll", "drone_pitch", "drone_yaw"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Plot 3D drone trajectory with orientation")
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help="Path to cam_ts_sync CSV (default: flight-01a-ellipse_cam_ts_sync.csv)",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_data(csv_path)
    n_frames = len(df)

    # Position and euler (radians)
    x = df["drone_x"].to_numpy()
    y = df["drone_y"].to_numpy()
    z = df["drone_z"].to_numpy()
    roll = df["drone_roll"].to_numpy()
    pitch = df["drone_pitch"].to_numpy()
    yaw = df["drone_yaw"].to_numpy()

    # Optional: use rotation matrix from CSV if present (more accurate)
    rot_cols = [f"drone_rot[{i}]" for i in range(9)]
    use_rot_matrix = all(c in df.columns for c in rot_cols)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Full trajectory (static)
    ax.plot(x, y, z, "k-", alpha=0.4, linewidth=0.8, label="Trajectory")

    # Drone body lines (will be updated by slider)
    drone_lines = get_drone_vertices(scale=0.08)
    frame_idx = 0

    def get_R(i):
        if use_rot_matrix:
            R = df.loc[i, rot_cols].values.reshape(3, 3)
            return np.asarray(R, dtype=float)
        return euler_to_rotation_matrix(roll[i], pitch[i], yaw[i])

    transformed = transform_drone_lines(drone_lines, get_R(frame_idx), x[frame_idx], y[frame_idx], z[frame_idx])
    artists = []
    for (lx, ly, lz) in transformed:
        art, = ax.plot(lx, ly, lz, "b-", linewidth=2)
        artists.append(art)

    # Current position marker
    pos_marker, = ax.plot([x[frame_idx]], [y[frame_idx]], [z[frame_idx]], "ro", markersize=6, label="Current")

    # Axis limits from trajectory
    margin = 0.5
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_zlim(z.min() - margin, z.max() + margin)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left")

    # Slider
    fig.subplots_adjust(bottom=0.12)
    ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        i = max(0, min(i, n_frames - 1))
        R = get_R(i)
        transformed = transform_drone_lines(drone_lines, R, x[i], y[i], z[i])
        for art, (lx, ly, lz) in zip(artists, transformed):
            art.set_data_3d(lx, ly, lz)
        pos_marker.set_data_3d([x[i]], [y[i]], [z[i]])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.suptitle("Drone trajectory and orientation (scroll slider to move in time)")
    plt.show()


if __name__ == "__main__":
    main()
