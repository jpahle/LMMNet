from lmmNet import lmmNet

import numpy as np
from scipy.integrate import odeint
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate cubic oscillator data and reconstruct the dynamics with lmmNet.')
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

def ml_f(x):
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
    errors = {} #store L1-norm of MAE at every grid point

    for M in [1, 2, 3, 4, 5]:
        for scheme in schemes:
            model = lmmNet(dt, X_train, M, scheme)
            model.train(N_Iter)

            errors[scheme + str(M)] = [np.sum(abs(ml_f(y) - f(y, None))) for y in X_train[0,:,:]]
            
    # save dictionary to file (important)
    with open('errors_T' + str(args.maxTime) + '_h' + str(args.stepSize) + '.pkl', 'wb') as file:
        pickle.dump(errors, file)
