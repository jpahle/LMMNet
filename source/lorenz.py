import numpy as np
import train_lmmNet

def f_lorenz(x,t):
    """
    The system of ODEs for the nonlinear Lorenz system with default parameters
    
    Arguments:
    x --  a list with three elements corresponding to the three variables
    t -- time
    
    Return:
    A numpy array containing the derivatives
    """
    
    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0

    r1 = sigma * (x[1] - x[0])
    r2 = x[0] * (rho - x[2]) - x[1]
    r3 = x[0] * x[1] - beta * x[2]

    return np.array([r1, r2, r3])

def simulate_default():

    # initial conditions
    x0 = np.array([-8.0, 7.0, 27])
    
    # time domain and step size
    t0, T, h = 0, 25, 0.01
    
    time_points, data = train_lmmNet.create_training_data(t0, T, h, f_lorenz, x0)
    
    return time_points, data