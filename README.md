# Adaptive Unscented Kalman Filter (AUKF) for Robust State Estimation in FPV Racing

# Description  

## Models ()
- use discrete-time nonlinear system to represent quadrotor kinematics 
- implement adaptation law that estimates instantaneous covariance as a function of sensor properties and dynamic flight regime 

## heterscedastic noise model

## Validation
1. Pybullet Simulation 
- environment: quadrotor autonomously navigates 3D racetrack using ground-truth based PID 
- sensor simulation: implemenent heterscedastic noise model to mimic MEMS sensor saturation 
    - sampled at 240 Hz 
- VIO: simulate vision-based position update by adding latency and gaussian white noise to ground truth position (sigma = 0.05m)
    - sampled at 30Hz 
- estimators: 
    - hover ukf: standard ukf with low Q & R
    - race ukf: standard ukf with static high Q & R
    - adaptive ukf  

2. Data Benchmark 
dataset: https://github.com/tii-racing/drone-racing-dataset/releases 
    - accelerometer/gyroscope logs 
- evaluation metrics: 
    - RMSE for position accuracy
    - NEES for filter consistency 
- plots 
    1. 3d trajectory comparison: 3D line plot of the track 
        - 3 plots to generate:
            - [ ] ground truth: perfect black line 
            - [ ] hover UKF: expect smooth but drifting wide on corners (overshoots)
            - [ ] race UKJ: expect tracking corners well but noisy "jitter" on straight sections
            - [ ] GP-AUKF: expect tight on corners, smooth on straights 
    2. position error vs g-force time-series:
        - axes:
            - x-axis: time (s)
            - y-axis:  
                - left: euclidean error (m)
                - right: g-force magnitude 
        - 3 plots to generate: 
            - [ ] hover UKF: expect error to spike synchronously with g-force peaks(turns)
            - [ ] race UKF: constant baseline error 
            - [ ] GP-AUKF: expect error to be independent of g-force magnitude 
    3. adaptive parameters (Q and R evolution)
        - axes: 
            - x-axis: time (s)
            - y-axis: value of diagonal of R (measurement noise)
        - 3 plots to generate: 
            - [ ] hover UKF: expect constant small R 
            - [ ] race UKF: expect constant large R 
            - [ ] GP-AUKF: expect race behavior in a turn, hover behavior in straightways 


