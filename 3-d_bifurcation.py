from lmmNet import lmmNet
import numpy as np
from scipy.integrate import odeint
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate data for Hopf system and reconstruct the dynamics with lmmNet.')
parser.add_argument('minTime', type=int,
                   help='Lower bound for the time domain')
parser.add_argument('maxTime', type=int,
                   help='Upper bound for the time domain')
parser.add_argument('stepSize', type=float, 
                   help='The mesh size')
parser.add_argument('--noise', action='store',default=0.0, type=float,
                   help='strenght of noise to be added to the training data (default: 0.00)')
parser.add_argument('--scheme', action='store',default='AM', type=str,
                   help='The family of LMM to use: choose from AM, AB, or BDF.')
parser.add_argument('--M', type=int, default=1,
                   help='the number of steps to use.')


def f(x,t):
    
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


def ml_f(x, t, model):
    """
    Define the derivatives learned by ML
    I think this is the best implementation, more robust than flatten()
    
    Args:
    x -- values for the current time point
    t -- time, dummy argument to conform with scipy's API
    model -- the learned ML model
    """
    return np.ravel(model.predict(x.reshape(1,-1)))


if __name__ == '__main__':
    
    global args
    args = parser.parse_args()
    
    # define the problem
    t0 = args.minTime
    T = args.maxTime
    h = args.stepSize
    time_points = np.arange(t0, T, h)

    # initial conditions to try
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
    
    ######################
    # create training data
    
    S = x0.shape[0] # number of trajectories
    N = time_points.shape[0] # number of time points (i.e. the grid size)
    D = x0.shape[1] # number of dimensions/species
    noise_strength = 0

    hopf_data = np.zeros((S, N, D))
    for k in range(0, S):
        hopf_data[k,:,:] = odeint(f, x0[k,:], time_points)

    hopf_data += noise_strength * hopf_data.std(1, keepdims=True) * np.random.randn(hopf_data.shape[0], hopf_data.shape[1], hopf_data.shape[2])
    
    #######################
    # train LmmNet
    hidden_layer_units = 256 # number of units for the hidden layer
    M = args.M # number of steps
    scheme = args.scheme # LMM scheme
    model = lmmNet(h, hopf_data, M, scheme, hidden_layer_units)

    N_Iter = 10000
    model.train(N_Iter)
    
    ########################
    # inference
    
    # initial conditions
    y0 = np.array([[-0.15,2,0],
                   [-0.05,2,0],

                   [.05,.01,0],
                   [.15,.01,0],
                   [.25,.01,0],
                   [.35,.01,0],
                   [.45,.01,0],
                   [.55,.01,0],

                   [.1,.01,0],
                   [.2,.01,0],
                   [.3,.01,0],
                   [.4,.01,0],
                   [.5,.01,0],
                   [.6,.01,0],

                   [0,.01,0],

                   [.05,2,0],
                   [.15,2,0],
                   [.25,2,0],
                   [.35,2,0],
                   [.45,2,0],
                   [.55,2,0],

                   [-0.2,2,0],
                   [-0.1,2,0],

                   [.1,2,0],
                   [.2,2,0],
                   [.3,2,0],
                   [.4,2,0],
                   [.5,2,0],
                   [.6,2,0],

                   [0,2,0]])
    
    
    hopf_pred = np.zeros((y0.shape[0], time_points.shape[0], y0.shape[1]))
    
    for k in range(0, y0.shape[0]):
        hopf_pred[k,:] = odeint(ml_f, y0[k,:], time_points, args=(model,))
    
    hopf = {}
    hopf['data'] = hopf_data
    hopf['pred'] = hopf_pred
    hopf['t'] = time_points

    # save to file
    with open('3-d_bifurcation.pkl', 'wb') as file:
            pickle.dump(hopf, file)