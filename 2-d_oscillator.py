from lmmNet import lmmNet

import numpy as np
from scipy.integrate import odeint
import argparse
import pickle
from utils import *

parser = argparse.ArgumentParser(description='Simulate cubic oscillator data and reconstruct the dynamics with lmmNet. Three family of schemes are trained separately, each using M = 1, 2, 3, 4, and 5.')
parser.add_argument('minTime', type=int,
                   help='Lower bound for the time domain')
parser.add_argument('maxTime', type=int,
                   help='Upper bound for the time domain')
parser.add_argument('stepSize', type=float, 
                   help='The mesh size')
parser.add_argument('--noise', action='store',default=0.0, type=float,
                   help='strenght of noise to be added to the training data (default: 0.00)')

    

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
