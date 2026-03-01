# ukf.py: contains the implementation of the standard UKF 
# noise covariances are constant 
import numpy as np 
import matplotlib.pyplot as plt 

class UKF():
    def __init__(self, P, Q, R, dt, ukf_params, quad_params, measurements): 
        
        # define variables 
        self.P = P # state covariance (changes with time)
        self.Q = Q   #process noise covariance
        self.R = R   #measurement noise covariance
        self.dt = dt # timestep 

        self.alpha = ukf_params['alpha'] #airframe structural resonance and dampening properties
        self.beta = ukf_params['beta'] #g-sensitivity coefficient of accelerometer and gyroscope axes 
        self.n = ukf_params['n'] #ukf params 
        self.lambd = ukf_params['lambda'] # ukf params 
        self.Wm = np.full(2 * self.n+1, 0.5/(self.n + self.lambd))
        self.Wm[0] = self.lambd / (self.n + self.lambd)
        self.Wc = np.full(2 * self.n+1, 0.5/(self.n + self.lambd))
        self.Wc[0] = self.lambd/(self.n + self.lambd) + (1 - self.alpha**2 + self.beta)
        
        self.mass = quad_params['mass']
        self.ct = quad_params['ct']       # thrust coefficient
        self.cd = quad_params['cd']       # drag coefficient
        self.L_arm = quad_params['L_arm'] # arm length
        self.g = 9.81

        self.measurements = measurements # from simulation data 
        
    def control(self, t):
        """
        u = prop speeds 
        """
        pass
    
    def unscented_transform(self, mu, cov):
        """
        performs the unscented transform on a mean and covariance and 
        returns the sigma points associated with the mean and covariance  
        """
        # matrix sqrt using eigenvalue decomposition 
        vals, vecs = np.linalg.eigh(cov)
        U = vecs @ np.diag(np.sqrt((self.n + self.lambd) * vals))

        sigma_points = np.zeros((2*self.n+1, self.n)) 
        sigma_points[0] = mu # first sigma point is the mean 
        for i in range(self.n):
            # unscented transform 
            sigma_points[i+1] = mu + U[:, i]
            sigma_points[self.n + i + 1] = mu - U[:, i]

        return sigma_points 

    def inv_unscented_transform(self, sigma_points):
        """
        given sigma points, returns the mean and covariance they were calculated from 
        
        no noise added
        """
        mu = np.dot(self.Wm, sigma_points)
        cov = np.zeros_like(self.Q) #same size as Q 
        for i in range(2 * self.n+1):
            diff = sigma_points[i] - mu
            cov += self.Wc[i] * np.outer(diff, diff)

        return mu, cov 

    def get_pred_sigma_points(self, prior_sigma_points, u):
        """
        propagates the prior sigma points through the nonlinear dynamics function 
        """
        pred_sigma_points = np.zeros((2 * self.n + 1, self.n))

        T = self.ct * np.sum(u**2)
        # body rates [p, q, r] from motor speed differences
        tau_p = self.L_arm * self.ct * (u[1]**2 - u[3]**2)
        tau_q = self.L_arm * self.ct * (u[2]**2 - u[0]**2)
        tau_r = self.cd * (u[0]**2 - u[1]**2 + u[2]**2 - u[3]**2)
        w_body = np.array([tau_p, tau_q, tau_r]) # simple body rate model

        for i in range(2 * self.n+1):
            p_prev = prior_sigma_points[i, 0:3]
            v_prev = prior_sigma_points[i, 3:6]
            q_prev = prior_sigma_points[i, 6:10] # [qw, qx, qy, qz]

            # propagate through f 
            p_k = p_prev + v_prev * self.dt # position update 
            
            # velocit update 
            R_mat = np.quaternions.quat_to_rot_mat(q_prev)
            accel_world = (R_mat @ np.array([0, 0, T / self.mass])) - np.array([0, 0, self.g])
            v_k = v_prev + accel_world * self.dt

            # quaternion update
            q_dot = 0.5 * np.quaternions.quat_multiply(q_prev, np.array([0, w_body[0], w_body[1], w_body[2]]))
            q_k = q_prev + q_dot * self.dt
            q_k = q_k / np.linalg.norm(q_k) # normalize to unite length 

            pred_sigma_points[i] = np.concatenate([p_k, v_k, q_k])

        return pred_sigma_points 

    def g(self, sigma_point):
        """
        maps 1 sigma point to the measurement space 
        """
        px, py, pz, vx, vy, vz, q1, q2, q3, q4 = sigma_point 
        q = np.array([q1, q2, q3, q4]) 
        
        # position meas from VIO 
        z_pos = np.array([px, py, pz]) 
        
        # acceleration meas from IMU 
        R_mat = np.quaternions.quat_to_rot_mat(q)
        gravity_world = np.array([0, 0, self.g])
        z_accel = R_mat.T @ gravity_world # Rotate gravity into body frame
        
        return np.concatenate([z_pos, z_accel])

    def get_measured_sigma_points(self, pred_sigma_points):
        """
        passes predicted sigma points through the measurement function 
        """
        meas_sigma_points = np.zeros((2 * self.n + 1, 6))

        for i in range(2 * self.n + 1):
            meas_sigma_points[i] = self.g(pred_sigma_points[i])
            
        return meas_sigma_points

    def prediction_step(self, mu_t, sigma_t, u_t): 
        """
        prediction step of the ukf 

        returns the predicted mean and covariance 
        """
        # unscented transform of the prior mean and covariance 
        prior_sigma_points = self.unscented_transform(mu_t, sigma_t)

        # propagate through nonlinear dynamics  
        pred_sigma_points = self.get_pred_sigma_points(prior_sigma_points, u_t)

        # inverse unscented transform + noise -> predicted mean and covariance
        pred_mu, pred_cov = self.inv_unscented_transform(pred_sigma_points)

        # add noise 
        pred_cov = pred_cov + self.Q 

        return pred_mu, pred_cov 
        
    def update_step(self, pred_mu, pred_cov, t):
        """
        update step of the ukf 

        returns the updated mean and covariance 
        """

        # unscented transform on predicted mean and covariance
        pred_sigma_points = self.unscented_transform(pred_mu, pred_cov)

        # propagate points through measurement function -> predicted sigma pt measurement
        meas_sigma_points = self.get_measured_sigma_points(pred_sigma_points)

        # get predicted mean, covariance, and cross-covariance measurements
        meas_y, meas_cov_y = self.inv_unscented_transform(meas_sigma_points)
        meas_cov_y += self.R # add sensor noise covariance 
        meas_cov_xy = np.zeros((self.n, 6)) 
        for i in range(2 * self.n + 1): # get cross covariance 
            state_diff = pred_sigma_points[i] - pred_mu
            meas_diff = meas_sigma_points[i] - meas_y
            meas_cov_xy += self.Wc[i] * np.outer(state_diff, meas_diff)
        
        # feed thru gaussian estimation equations -> updated mean and cov 
        K = meas_cov_xy @ np.linalg.inv(meas_cov_y)
        updated_mu = pred_mu + K @ (self.measurements[t] - meas_y)  
        updated_cov = pred_cov - K @ meas_cov_y @ K.T

        return updated_mu, updated_cov 
         

    def simulate(self, pose0, N, u):
        """
        simulates the ukf for N timesteps given the initial condition and controls 
        """
        poses_history = [pose0]
        measurements_history = []
        true_controls_history = []
        mu_t = pose0 # unpack initial condition
        cov_t = self.P 

        for i in range(N):
            # prediction 
            pred_mu, pred_cov = self.prediction_step(mu_t, cov_t)
            
            # update 
            updated_mu, updated_cov = self.update_step(pred_mu, pred_cov)

            poses_history.append(mu_t)

        return np.array(poses_history)

    def plot_position_history(self, poses, show_plot=True):
        """
        Plot the position history of the quadcoptor. 
        """
        # check that we have only one run
        assert poses.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {poses.ndim} dimensions."
        
        # plot the pose history and the sensor measurements
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("pose History")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # plot the initial pose with a black star at the highest z-order
        plt.scatter(poses[0, 0], poses[0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial pose")

        # plot the position history 
        plt.plot(poses[:, 0], poses[:, 1], poses[:, 2] label="pose")

        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig


    def plot_quaternion_history(self, poses, show_plot=True):
        """
        """
        pass 
    
    def plot_control_history(self, controls, show_plot=True):
        # check that we have only one run
        assert controls.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {controls.ndim} dimensions."
        
        # plot the control history
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Control History")
        plt.xlabel("Time (s)")
        plt.ylabel("s (m/s), phi (rad/s)")

        # Plot the control history
        plt.plot(controls[:, 0], label="Speed Input")
        plt.plot(controls[:, 1], label="Rotation Rate Input")
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()
        
        return fig


    def plot_measurement_history(self, measurements, show_plot=True):
        # check that we have only one run
        assert measurements.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {measurements.ndim} dimensions."
        
        # plot the measurement history
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Measurement History")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance to Base Station (m)")

        # plot the measurement history
        for station_ind in range(self.num_stations):
            plt.plot(measurements[:, station_ind], 
                     label=f"Distance to Base Station {station_ind}")
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if show_plot:
            plt.show()
        
        return fig


