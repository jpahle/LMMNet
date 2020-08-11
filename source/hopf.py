import numpy as np
from scipy.integrate import odeint

def f_hopf(x,t):
    
    """
    System of ODEs for the Hopf normal form
    
    Arguments:
    x -- a list of three elements corresponding to the initial values of the three species: the constant bifurcation parameter and the two variables in the system.
    
    t -- time
    
    Returns:
    a list containing the derivatives
    """
    
    mu = x[0] # mu is constant
    omega = 1
    sigma = 1

    r1 = 0
    r2 = mu * x[1] - omega * x[2] - sigma * x[1] * (x[1]**2 + x[2]**2)
    r3 = omega * x[1] + mu * x[2] - sigma * x[2] * (x[1]**2 + x[2]**2)

    return np.array([r1, r2, r3])

def simulate_default():
    """
    Simulate Hopf data with default settings
    """
    
    # define the time domain
    tfirst, tlast, step = 0, 75, 0.1
    time_points = np.arange(tfirst, tlast, step)
    
    # initial conditions for training
    x0 = np.array([[-0.15,.01,0],
                   [-0.05,.01,0],

                   [.05,.01,0],
                   [.15,.01,0],
                   [.25,.01,0],
                   [.35,.01,0],
                   [.45,.01,0],
                   [.55,.01,0],

                   [-0.15,2,0],
                   [-0.05,2,0],

                   [.05,2,0],
                   [.15,2,0],
                   [.25,2,0],
                   [.35,2,0],
                   [.45,2,0],
                   [.55,2,0]])
    
    
    
    S = x0.shape[0] # number of trajectories
    N = time_points.shape[0] # number of time points (i.e. the grid size)
    D = x0.shape[1] # number of dimensions/species
    noise_strength = 0

    hopf_data = np.zeros((S, N, D))
    for k in range(0, S):
        hopf_data[k,:,:] = odeint(f_hopf, x0[k,:], time_points)

    hopf_data += noise_strength * hopf_data.std(1, keepdims=True) * np.random.randn(hopf_data.shape[0], hopf_data.shape[1], hopf_data.shape[2])
    
    
    return time_points, hopf_data
    
    