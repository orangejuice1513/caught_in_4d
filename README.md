# Adaptive Unscented Kalman Filter (AUKF) for Robust State Estimation in FPV Racing

Autonomous Drone Racing (ADR) pushes quadrotor trajectories to highly stiff and nonlinear flight profiles, often exceeding 5Gs in turns and 100 mph in straightaways. In these extremes, low-cost MEMS Inertial Measurement Units (IMUs) exhibit **heteroscedastic noise behavior**—sensor noise is not constant, but rather increases drastically with motor RPM (vibration) and G-loading.

This repository implements a **Physics-Informed Adaptive Unscented Kalman Filter (AUKF)** that learns the noise environment in real-time to maintain robust state estimation during extreme maneuvers.



## 🛑 The Problem: The Static Tuning Trade-Off
Traditional Unscented Kalman Filters (UKF) rely on a static Measurement Noise Covariance matrix ($R$), creating a fatal trade-off in drone racing:
* **Tuning for Hover (Low $R$):** The filter trusts the IMU heavily. During a high-G turn, vibration-induced noise is interpreted as valid motion, causing the state estimate to jitter and the control loop to destabilize.
* **Tuning for Racing (High $R$):** The filter distrusts the IMU. In low-dynamic sections (e.g., precision gate entry), the filter ignores subtle sensor cues, leading to drift and gate collisions.

## 🚀 Our Solution: Physics-Informed Adaptation
Unlike existing adaptive methods that react to statistical errors *after* performance degrades, our AUKF introduces a **predictive, physics-informed adaptation layer**. By explicitly modeling measurement noise covariance ($\mathbf{R}$) as a function of control input (RPM) and inertial loading (G-force), the filter anticipates noise shifts *before* state corruption occurs.

### The Heteroscedastic Noise Model
We consider a discrete-time nonlinear system representing quadrotor kinematics, formulated using unit quaternions to avoid gimbal lock:

$$\mathbf{x}_{k} = f(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}) + \mathbf{w}_{k-1}$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

We model the measurement noise $\mathbf{v}_k$ as a non-stationary Gaussian process where the instantaneous covariance $\mathbf{R}_k$ is a composite function of static sensor properties and the dynamic flight regime:

$$\mathbf{R}_k = \mathbf{R}_{\text{floor}} + \mathbf{R}_{\text{vibe}}(\Omega_k) + \mathbf{R}_{\text{load}}(\mathbf{a}_k)$$

1. **RPM-Induced Vibration:** Modeled as proportional to the total kinetic energy of the rotors (frequency squared).
   $$\mathbf{R}_{\text{vibe}}(\Omega_k) = \alpha \cdot \text{diag} \left( \sum_{i=1}^{4} \Omega_{i,k}^2 \right) \cdot \mathbf{I}_{3 \times 3}$$

2. **G-Loading Sensitivity:** Accounts for MEMS sensor non-linearities during high-acceleration maneuvers by scaling with the specific force magnitude.
   $$\mathbf{R}_{\text{load}}(\mathbf{a}_k) = \beta \cdot \|\mathbf{a}_k\| \cdot \mathbf{I}_{3 \times 3}$$



## 🧪 Validation & Benchmarking  

We evaluate the proposed AUKF across two phases:

### Phase 1: High-Fidelity Simulation (PyBullet)
* **Environment:** Autonomous 3D racing track navigation (split-S maneuvers, high-speed banks) using a Ground Truth-based PID controller.
* **Sensors:** * IMU (240Hz): Heteroscedastic noise scaling linearly with G-load and motor RPM.
  * VIO (30Hz): Simulated vision updates with Gaussian white noise ($\sigma = 0.05m$) and 30ms latency.
* **Tested Estimators:** Hover UKF, Race UKF, and our Physics-Informed Adaptive UKF.

### Phase 2: Real-World Benchmarking
Replaying the **TII Drone Racing Dataset (Blackbird/UZH-FPV)** through our filters to test against real aerodynamic disturbances (prop wash, ground effect) flown by human pilot MinChan Kim.

### Key Metrics
* **RMSE:** Root Mean Square Error for position accuracy.
* **NEES:** Normalized Estimation Error Squared to evaluate filter consistency.



## 💻 Project Structure
```text
├── data/                  # datasets 
├── docs/                  # project proposals 
├── scripts/               # utility scripts for data processing
├── sim/                   # PyBullet simulation environment
├── src/
│   ├── filters/           # implementations of standard and adaptive UKFs
│   ├── models/            # quadroter kinematics and noise models
│   └── main.py            # main execution script
├── requirements.txt       # python dependencies
└── README.md


```

# Authors 
- Julia Jiang - Electrical Engineering, Stanford University
- Koichi Kimoto - Aeronautics and Astronautics, Stanford University
- Artash Nath - Computer Science, Stanford University
- Kelvin Nguyen - Computer Science, Stanford University