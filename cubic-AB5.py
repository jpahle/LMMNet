from lmmNet import lmmNet

import numpy as np
from scipy.integrate import odeint
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate cubic oscillator data and reconstruct the dynamics with lmmNet AB5, using the Adams Bashforth scheme with 5 steps.')
parser.add_argument('minTime', type=int,
                   help='Lower bound for the time domain')
parser.add_argument('maxTime', type=int,
                   help='Upper bound for the time domain')
parser.add_argument('stepSize', type=float, 
                   help='The mesh size')
parser.add_argument('--noise', action='store',default=0.0, type=float,
                   help='strenght of noise to be added to the training data (default: 0.00)')
parser.add_argument('--skip', action='store',default=1, type=int,
                   help='number of observation points to skip when generating training data (default: 1)')


def f(x,t):
    """
    Return the derivatives (RHS of the ODE)
    This is a linear system with the form f = A x
    Args:
    x -- a 2 x 1 vector of measurements
    """
    A = np.array([[-0.1, 2], [-2,-0.1]]) # 2 x 2

    return np.ravel(np.matmul(A,x.reshape(-1, 1)**3))

def ml_f(x, model):
    """
    Define the derivatives (RHS of the ODE) learned by ML
    I think this is the best implementation (more robust than flatten())
    """
    return np.ravel(model.predict(x.reshape(1,-1)))
    

if __name__ == "__main__":
    
    global args
    args = parser.parse_args()
    
    # create time points
    time_points = np.arange(args.minTime, args.maxTime, args.stepSize) # 100 grid points

    # specify initial conditions
    x0 = np.array([2,0])

    simulated_x = odeint(f, x0, time_points)
    
    # create training data
    noise = args.noise #strength of the noise

    skip = args.skip
    dt = time_points[skip] - time_points[0]
    X_train = simulated_x[0::skip,:]
    X_train = X_train + noise * X_train.std(0) * np.random.randn(X_train.shape[0], X_train.shape[1])

    X_train = np.reshape(X_train, (1,X_train.shape[0],X_train.shape[1]))
    
    # right now the layers are hardcoded.
    schemes = ['AB', 'BDF', 'AM']
    N_Iter = 10000
    errors = {}
    errors['f'] = [f(y, None) for y in X_train[0,:,:]]
    model = lmmNet(dt, X_train, 5, 'BDF')
    model.train(N_Iter)
    errors['e'] = [np.sum(abs(ml_f(y, model) - f(y, None))) for y in X_train[0,:,:]]
    errors['ml_f'] = [ml_f(y, model) for y in X_train[0,:,:]]
            
    # save dictionary to file (important)
    with open('errors_AB5.pkl', 'wb') as file:
        pickle.dump(errors, file)
