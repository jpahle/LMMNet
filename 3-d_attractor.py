from lmmNet import lmmNet
import numpy as np
from scipy.integrate import odeint
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate 3-D Lorenz attractor data and reconstruct the dynamics with lmmNet.')
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

if __name__ == "__main__":
    
    global args
    args = parser.parse_args()
    
    # initial condition
    x0 = np.array([-8.0, 7.0, 27])

    t0 = args.minTime # start time
    T = args.maxTime # end time
    h = args.stepSize # step size

    # generate data
    time_points = np.arange(t0, T, h)
    lorenz_data = odeint(f, x0, time_points)
    
    # add Gaussian noise scaled by standard deviation, for every one of the three dimensions
    noise_strength = args.noise
    lorenz_data += noise_strength * lorenz_data.std(0) * np.random.randn(lorenz_data.shape[0], lorenz_data.shape[1])

    lorenz_data = np.reshape(lorenz_data, (1,lorenz_data.shape[0], lorenz_data.shape[1]))
    
    hidden_layer_units = 256 # number of units for the hidden layer
    M = args.M # number of steps
    scheme = args.scheme # LMM scheme
    model = lmmNet(h, lorenz_data, M, scheme, hidden_layer_units)

    N_Iter = 10000
    model.train(N_Iter)
    
    lorenz_pred = odeint(ml_f, x0, time_points, args=(model,))
    
    lorenz = {}
    lorenz['data'] = lorenz_data
    lorenz['pred'] = lorenz_pred
    lorenz['f'] = [f(y, None) for y in lorenz_data[0,:,:]]
    lorenz['ml_f'] = [ml_f(y, None, model) for y in lorenz_data[0,:,:]]
    lorenz['t'] = time_points

    # save to file
    with open('3-d_attractor.pkl', 'wb') as file:
            pickle.dump(lorenz, file)