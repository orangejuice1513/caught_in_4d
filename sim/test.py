from sim.env import DroneEnv, SimConfig, DroneConfig
from sim.controller import GeometricController, ReferenceState
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

env = DroneEnv(
    sim_cfg=SimConfig(dt=1/240, gui=True),
    drone_cfg=DroneConfig(
        urdf_path=str(ROOT / "assets" / "quad.urdf"),
        expected_mass_kg=1.35,
        start_pos_w=np.array([0.0, 0.0, 0.8]),
    ),
)

truth = env.reset()
controller = GeometricController.from_env(env)

for k in range(9400):
    truth = env.get_truth_state()

    ref = ReferenceState(
        pos_w=np.array([0.0, 0.0, 1.5]),
        vel_w=np.zeros(3),
        acc_w=np.zeros(3),
        yaw=0.0,
        yaw_rate=0.0,
    )

    rpm_cmd = controller.compute_rpm(truth, ref, dt=env.sim_cfg.dt)
    truth = env.step(rpm_cmd)

    if k % 120 == 0:
        print(
            f"t={truth.t:.2f} "
            f"pos={truth.pos_w} "
            f"vel={truth.vel_w}"
        )

env.close()