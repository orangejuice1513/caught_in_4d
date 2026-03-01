# ukf.py: contains the implementation of the standard UKF 
# noise covariances are constant 
import numpy as np 
import matplotlib.pyplot as plt 

class UKF():
    def __init__(self, P, Q, R, dt, sigma, alpha, beta, n, lambd, quad_params): 
        
        # define variables 
        self.P = P # state covariance (changes with time)
        self.Q = Q   #process noise covariance
        self.R = R   #measurement noise covariance
        self.dt = dt # timestep 

        self.sigma = 2 # unscented transform param
        self.alpha = 1 #airframe structural resonance and dampening properties
        self.beta = 1 #g-sensitivity coefficient of accelerometer and gyroscope axes 
        self.n = 2 #ukf params 
        self.lambd = 2 # ukf params 
        self.Wm = np.full(2 * self.n+1, 0.5/(self.n + self.lambd))
        self.Wm[0] = self.lambd / (self.n + self.lambd)
        self.Wc = np.full(2 * self.n+1, 0.5/(self.n + self.lambd))
        self.Wc[0] = self.lambd/(self.n + self.lambd) + (1 - self.alpha**2 + self.beta)
        
        self.mass = quad_params['mass']
        self.ct = quad_params['ct']       # thrust coefficient
        self.cd = quad_params['cd']       # drag coefficient
        self.L_arm = quad_params['L_arm'] # arm length
        self.g = 9.81

        
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
        """
        mu = np.dot(self.Wm, sigma_points)
        cov = self.Q.copy()
        for i in range(2 * self.n+1):
            diff = sigma_points[i] - mu
            cov_pred += self.Wc[i] * np.outer(diff, diff)

        return mu, cov 


    def get_pred_sigma_points(self, prior_sigma_points):
        """
        propagates the prior sigma points through the nonlinear dynamics function 
        """
        pred_sigma_points = np.array((2 * self.n + 1, self.n))

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

    def prediction_step(self, mu_t, sigma_t, t): 
        """
        prediction step of the ukf 

        returns the predicted mean and covariance 
        """
        # unscented transform of the prior mean and covariance 
        prior_sigma_points = self.unscented_transform(self, mu_t, sigma_t)

        # propagate through nonlinear dynamics  
        pred_sigma_points = self.get_pred_sigma_points(self, prior_sigma_points)

        # inverse unscented transform + noise -> predicted mean and covariance
        pred_mu, pred_cov = self.inv_unscented_transform(self, pred_sigma_points)

        return pred_mu, pred_cov 
        
    def update_step(self, pred_mu, pred_cov):
        """
        update step of the ukf 

        returns the updated mean and covariance 
        """

    def dynamics_step(self, pose, t, u):
        """
        given P_t, computes P_t+1 
        """
        px, py, pz, vx, vy, vz, q1, q2, q3, q4 = pose  # unpack the pose 

        # compute new pose 
        pass 

    def measurement_step(self, pose, t, u):
        pass 

    def simulate(self, pose0, N, u):
        for i in range(N):
            pass 

    def plot_position_history(self, poses, show_plot=True):
        """
        Plot the position history of the quadcoptor. 
        """

    def plot_quaternion_history(self, poses, show_plot=True):
        """
        """
        pass 
    
    def plot_control_history(self, controls, show_plot=True):
        pass 

    def plot_measurement_history(self, measurements, show_plot=True):
        pass

