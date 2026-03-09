from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot UKF replay results against truth.")
    p.add_argument(
        "--sim-log",
        type=str,
        default=None,
        help="Path to sim_run.npz. Default: <repo>/logs/sim_run.npz",
    )
    p.add_argument(
        "--filter-log",
        type=str,
        default=None,
        help="Path to filter_results.npz. Default: <repo>/logs/filter_results.npz",
    )
    return p


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def maybe_load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def compute_pos_error(est_pos: np.ndarray, truth_pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    err = np.full(len(truth_pos), np.nan, dtype=float)
    if np.any(valid):
        diff = est_pos[valid] - truth_pos[valid]
        err[valid] = np.linalg.norm(diff, axis=1)
    return err


def binned_mean(x: np.ndarray, y: np.ndarray, nbins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])

    edges = np.linspace(np.min(x), np.max(x), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < nbins - 1 else x <= edges[i + 1])
        if np.any(m):
            means[i] = float(np.mean(y[m]))

    keep = np.isfinite(means)
    return centers[keep], means[keep]


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    sim_path = Path(args.sim_log).expanduser().resolve() if args.sim_log else repo_root / "logs" / "sim_run.npz"
    filt_path = Path(args.filter_log).expanduser().resolve() if args.filter_log else repo_root / "logs" / "filter_results.npz"

    if not sim_path.exists():
        raise FileNotFoundError(f"Missing sim log: {sim_path}")
    if not filt_path.exists():
        raise FileNotFoundError(f"Missing filter log: {filt_path}")

    sim = load_npz(sim_path)
    filt = load_npz(filt_path)
    filt_meta = maybe_load_json(filt_path.with_suffix(".json"))

    out_dir = filt_path.parent

    # Truth
    t = sim["t"]
    truth_pos = sim["truth_pos_w"]
    truth_g = sim["truth_g_load"]

    # Filter outputs
    hover_pos = filt["hover_pos_w"]
    race_pos = filt["race_pos_w"]
    adaptive_pos = filt["adaptive_pos_w"]

    hover_valid = filt["hover_valid"].astype(bool)
    race_valid = filt["race_valid"].astype(bool)
    adaptive_valid = filt["adaptive_valid"].astype(bool)

    hover_err = compute_pos_error(hover_pos, truth_pos, hover_valid)
    race_err = compute_pos_error(race_pos, truth_pos, race_valid)
    adaptive_err = compute_pos_error(adaptive_pos, truth_pos, adaptive_valid)

    # Optional adaptive traces
    adaptive_pos_var = filt["adaptive_pos_var"] if "adaptive_pos_var" in filt else None
    adaptive_vel_var = filt["adaptive_vel_var"] if "adaptive_vel_var" in filt else None
    adaptive_att_var = filt["adaptive_att_var"] if "adaptive_att_var" in filt else None

    # --------------------------------------------------
    # 1) 3D trajectory comparison
    # --------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2], linewidth=2.4, label="Ground truth")
    if np.any(hover_valid):
        ax.plot(hover_pos[hover_valid, 0], hover_pos[hover_valid, 1], hover_pos[hover_valid, 2], linewidth=1.6, label="Hover UKF")
    if np.any(race_valid):
        ax.plot(race_pos[race_valid, 0], race_pos[race_valid, 1], race_pos[race_valid, 2], linewidth=1.6, label="Race UKF")
    if np.any(adaptive_valid):
        ax.plot(adaptive_pos[adaptive_valid, 0], adaptive_pos[adaptive_valid, 1], adaptive_pos[adaptive_valid, 2], linewidth=1.8, label="Adaptive UKF")

    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    save_fig(fig, out_dir / "filter_trajectory_3d.png")

    # --------------------------------------------------
    # 2) Position error vs time with G-load overlay
    # --------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(t, hover_err, linewidth=1.5, label="Hover UKF")
    ax1.plot(t, race_err, linewidth=1.5, label="Race UKF")
    ax1.plot(t, adaptive_err, linewidth=1.7, label="Adaptive UKF")
    ax1.set_title("Position Error vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position error (m)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, truth_g, linestyle="--", linewidth=1.1, alpha=0.8, label="G-load")
    ax2.set_ylabel("G-load")

    save_fig(fig, out_dir / "filter_error_vs_time.png")

    # --------------------------------------------------
    # 3) Position error vs G-load
    # --------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    configs = [
        ("Hover UKF", hover_err),
        ("Race UKF", race_err),
        ("Adaptive UKF", adaptive_err),
    ]

    for ax, (title, err) in zip(axs, configs):
        mask = np.isfinite(err) & np.isfinite(truth_g)
        ax.scatter(truth_g[mask], err[mask], s=6, alpha=0.18)
        cx, cy = binned_mean(truth_g, err, nbins=20)
        if len(cx) > 0:
            ax.plot(cx, cy, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("G-load")
        ax.grid(alpha=0.2)

    axs[0].set_ylabel("Position error (m)")
    fig.suptitle("Position Error vs G-load", y=1.02)
    save_fig(fig, out_dir / "filter_error_vs_gload.png")

    # --------------------------------------------------
    # 4) Adaptive process-noise evolution
    # --------------------------------------------------
    if adaptive_pos_var is not None and adaptive_vel_var is not None and adaptive_att_var is not None:
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(t, adaptive_pos_var, linewidth=1.5)
        axs[0].set_ylabel("Pos var")
        axs[0].set_title("Adaptive Process Noise Evolution")

        axs[1].plot(t, adaptive_vel_var, linewidth=1.5)
        axs[1].set_ylabel("Vel var")

        axs[2].plot(t, adaptive_att_var, linewidth=1.5)
        axs[2].set_ylabel("Att var")
        axs[2].set_xlabel("Time (s)")

        save_fig(fig, out_dir / "adaptive_q_evolution.png")

    # --------------------------------------------------
    # 5) Top-down XY path + altitude tracking
    # --------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    axs[0].plot(truth_pos[:, 0], truth_pos[:, 1], linewidth=2.2, label="Ground truth")
    if np.any(hover_valid):
        axs[0].plot(hover_pos[hover_valid, 0], hover_pos[hover_valid, 1], linewidth=1.4, label="Hover UKF")
    if np.any(race_valid):
        axs[0].plot(race_pos[race_valid, 0], race_pos[race_valid, 1], linewidth=1.4, label="Race UKF")
    if np.any(adaptive_valid):
        axs[0].plot(adaptive_pos[adaptive_valid, 0], adaptive_pos[adaptive_valid, 1], linewidth=1.6, label="Adaptive UKF")
    axs[0].set_title("Top-Down Path")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].axis("equal")
    axs[0].legend()

    axs[1].plot(t, truth_pos[:, 2], linewidth=2.2, label="Ground truth")
    if np.any(hover_valid):
        axs[1].plot(t[hover_valid], hover_pos[hover_valid, 2], linewidth=1.4, label="Hover UKF")
    if np.any(race_valid):
        axs[1].plot(t[race_valid], race_pos[race_valid, 2], linewidth=1.4, label="Race UKF")
    if np.any(adaptive_valid):
        axs[1].plot(t[adaptive_valid], adaptive_pos[adaptive_valid, 2], linewidth=1.6, label="Adaptive UKF")
    axs[1].set_title("Altitude Tracking")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Z (m)")
    axs[1].legend()

    save_fig(fig, out_dir / "filter_xy_and_altitude.png")

    # --------------------------------------------------
    # 6) Error summary bar chart
    # --------------------------------------------------
    rmse_hover = float(np.sqrt(np.nanmean(hover_err ** 2)))
    rmse_race = float(np.sqrt(np.nanmean(race_err ** 2)))
    rmse_adaptive = float(np.sqrt(np.nanmean(adaptive_err ** 2)))

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Hover UKF", "Race UKF", "Adaptive UKF"]
    vals = [rmse_hover, rmse_race, rmse_adaptive]
    bars = ax.bar(labels, vals)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Filter Position RMSE")

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.3f}", ha="center", va="bottom")

    save_fig(fig, out_dir / "filter_rmse_bar.png")

    # --------------------------------------------------
    # Console summary
    # --------------------------------------------------
    print("\n--- Filter Plot Summary ---")
    print(f"Hover RMSE:    {rmse_hover:.4f} m")
    print(f"Race RMSE:     {rmse_race:.4f} m")
    print(f"Adaptive RMSE: {rmse_adaptive:.4f} m")
    if filt_meta:
        print("\nJSON summary:")
        print(json.dumps(filt_meta, indent=2))


if __name__ == "__main__":
    main()