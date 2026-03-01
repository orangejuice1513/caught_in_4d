# measurement_noise.py: contains the heteroscedastic measurement noise model 
import numpy as np 

def measurement_noise_cov(RPM_k, a_k):
    """
    RPM_k: 4x4 diagonal matrix of motor RPMs 
    a_k: 1x3 acceleration vector 
    """
    
    R_vibe = alpha * np.diag(sum())
    
    return