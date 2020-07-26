import numpy as np
import train_lmmNet

def f_cubic(x,t):
    """
    Return the derivatives (RHS of the ODE)
    This is a linear system with the form f = A x
    Args:
    x -- a 2 x 1 vector of measurements
    """
    A = np.array([[-0.1, 2], [-2,-0.1]]) # 2 x 2

    return np.ravel(np.matmul(A,x.reshape(-1, 1)**3))

def simulate_default():
    """
    Simulate the default 2-D damped harmonic oscillator cubic dynamics
    """
    
    # time domain
    t0, T, h = 0, 25, 0.01
    
    x0 = np.array([2,0]) # initial conditions -- default cubic problem
    
    time_points, cubic_data = train_lmmNet.create_training_data(t0, T, h, f_cubic, x0)
    
    return time_points, cubic_data

def simulate_custom(tstart=0, tend=25, step=.01, xinit=2, yinit=0):
    """
    Simulate the default 2-D damped harmonic oscillator cubic dynamics
    """
    
    # time domain
    t0, T, h = tstart, tend, step
    
    x0 = np.array([xinit,yinit]) # initial conditions -- default cubic problem
    
    time_points, cubic_data = train_lmmNet.create_training_data(t0, T, h, f_cubic, x0)
    
    return time_points, cubic_data