# Adaptive Unscented Kalman Filter (AUKF) for Robust State Estimation in FPV Racing

## Description  
- Use discrete-time nonlinear system to represent quadrotor kinematics 
- Implement adaptation law that estimates instantaneous covariance as a function of sensor properties and dynamic flight regime 

## Models 
- **State**: The state vector $\mathbf{x} \in \mathbb{R}^{10}$ comprises position, velocity, and the attitude quaternion to avoid gimbal lock:
  
  $$\mathbf{x} = [\mathbf{p}^\top, \mathbf{v}^\top, \mathbf{q}^\top]^\top$$
  
  *(Note: attitude quaternion represents the rotation from the body frame to the inertial frame).*
  - [ ] TODO: `state.py`: contains the state data structure 

- **Process Noise**: Represents the unmodeled dynamics and disturbances in the quadrotor kinematics.
  - **Covariance**: The static process noise matrix (Q).
  - [ ] TODO: `process_noise.py`: contains process noise functions 

- **Measurement**: The sensor readings from the MEMS IMU and the Computer Vision pipeline.
  - [ ] TODO: `measurement.py`: contains the measurement data

- **Measurement Noise**: Modeled as a non-stationary Gaussian process, capturing the heteroscedastic nature of high-dynamic flight:
  
  $$\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R}_k)$$
  
  - **Covariance**: The instantaneous covariance $\mathbf{R}_k = g(\text{RPM}_k, \mathbf{a}_k)$ is an adaptation law estimating noise as a composite function of static properties and the dynamic flight regime:
    
    $$\mathbf{R}_k = \mathbf{R}_{\text{floor}} + \mathbf{R}_{\text{vibe}}(\Omega_k) + \mathbf{R}_{\text{load}}(\mathbf{a}_k)$$
    
    - **RPM-Induced Vibration**: Given the rotational velocity of the four motors $\Omega_{i,k}$ (in RPM), the vibration noise injected into the IMU is modeled as proportional to the total kinetic energy of the rotors (proportional to frequency squared):
      
      $$\mathbf{R}_{\text{vibe}}(\Omega_k) = \alpha \cdot \text{diag} \left( \sum_{i=1}^{4} \Omega_{i,k}^2 \right) \cdot \mathbf{I}_{3 \times 3}$$
      
      *(Where α is a constant coefficient characterizing the airframe's structural resonance and dampening properties).*
    
    - **G-Loading Sensitivity**: To account for MEMS sensor non-linearities during high-acceleration maneuvers, the covariance scales with the magnitude of the specific force (linear acceleration) $\|\mathbf{a}_k\|$:
      
      $$\mathbf{R}_{\text{load}}(\mathbf{a}_k) = \beta \cdot \|\mathbf{a}_k\| \cdot \mathbf{I}_{3 \times 3}$$
      
      *(Where β is the empirical G-sensitivity coefficient of the accelerometer and gyroscope axes).*
    - [ ] TODO: `measurement_noise.py`: contains the structures and functions for measurement noise 
    
## Filter Implementation
Task: Make classes of standard UKF and adaptive UKF  
- [ ] Standard UKF 
- [ ] Adaptive UKF 

## Validation
- [ ] PyBullet Simulation 
    - **Environment:** Quadrotor autonomously navigates 3D racetrack using ground-truth based PID 
    - **Sensor simulation:** Implement heteroscedastic noise model to mimic MEMS sensor saturation 
        - Sampled at 240 Hz 
    - **VIO:** Simulate vision-based position update by adding latency and Gaussian white noise to ground truth position (σ = 0.05m)
        - Sampled at 30Hz 
- **Estimators:** - [ ] Hover UKF: standard UKF with low Q & R
    - [ ] Race UKF: standard UKF with static high Q & R
    - [ ] Adaptive UKF  

### Data Benchmark 
Dataset: https://github.com/tii-racing/drone-racing-dataset/releases 
- Accelerometer/gyroscope logs 
- **Evaluation metrics:** - RMSE for position accuracy
    - NEES for filter consistency 
- **Plots** 1. **3D trajectory comparison:** 3D line plot of the track 
        - 4 plots to generate:
            - [ ] Ground truth: perfect black line 
            - [ ] Hover UKF: expect smooth but drifting wide on corners (overshoots)
            - [ ] Race UKF: expect tracking corners well but noisy "jitter" on straight sections
            - [ ] GP-AUKF: expect tight on corners, smooth on straights 
    2. **Position error vs g-force time-series:**
        - Axes:
            - X-axis: time (s)
            - Y-axis:  
                - Left: euclidean error (m)
                - Right: g-force magnitude 
        - 3 plots to generate: 
            - [ ] Hover UKF: expect error to spike synchronously with g-force peaks (turns)
            - [ ] Race UKF: constant baseline error 
            - [ ] GP-AUKF: expect error to be independent of g-force magnitude 
    3. **Adaptive parameters (Q and R evolution):**
        - Axes: 
            - X-axis: time (s)
            - Y-axis: value of diagonal of R (measurement noise)
        - 3 plots to generate: 
            - [ ] Hover UKF: expect constant small R 
            - [ ] Race UKF: expect constant large R 
            - [ ] GP-AUKF: expect race behavior in a turn, hover behavior in straightaways