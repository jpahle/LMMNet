import numpy as np
import train_lmmNet

def f_linear(x,t):
    """
    Return the derivatives (RHS of the ODE)
    This is a linear system with the form f = A x
    
    Args:
    x -- a 2 x 1 vector of measurements
    """
    A = np.array([[-0.1, -2, 0], [2,-0.1, 0], [0, 0, -0.3]]) # 3 x 3

    return np.ravel(np.matmul(A,x.reshape(-1, 1)))

def simulate_default():
    """
    Simulate the 3-D oscillator with linear dynamics
    using the default parameters
    """
    
    tfirst, tlast, step = 0, 50, 0.01
    
    x0 = np.array([2,0,1]) #initial values
    
    time_points, data = train_lmmNet.create_training_data(tfirst, tlast, step, f_linear, x0)
    
    return time_points, data
    

def simulate_custom(noise=0, step=0.01, xinit=2, yinit=0, zinit=1):
    """
    Simulate the 3-D oscillator with linear dynamics
    using the default parameters
    """
    
    tfirst, tlast = 0, 50
    
    x0 = np.array([xinit, yinit, zinit]) #initial values
    
    time_points, data = train_lmmNet.create_training_data(tfirst, tlast, step, f_linear, x0, noise_strength=noise)
    
    return time_points, data