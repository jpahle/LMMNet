from lmmNet import lmmNet
import numpy as np
from scipy.integrate import odeint
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate 7-D Yeast Glycolysis Oscillator and reconstruct the dynamics with lmmNet.')
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


def f(x, params, N = 1., A = 4., phi = 0.1):
    """
    ODE model for yeast glycolytic oscillations
    Default parameter values are taken from https://www.sciencedirect.com/science/article/pii/S0301462203001911
    
    Args:
    * x -- array of concentrations of the 7 biochemical species
    * parameters -- dictionary of parameters
    * N -- total concentration of NAD+ and NADH
    * A -- total concentration of ADP and ATP
    * phi -- ratio of the total cellular volume to the extracellular volume
    """
    Jin = params['Jin']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    k5 = params['k5']
    k6 = params['k6']
    k = params['k']
    kappa = params['kappa'] # rate constant related to the permeability of the membrane
    q = params['q'] # cooperativity coefficient of the ATP inhibition
    KI = params['KI'] #inhibition constant
    
    s1, s2, s3, s4, s5, s6, s7 = x
    
    ds1 = Jin - k1 * s1 * s6/(1 + (s6/KI)**q)
    ds2 = 2 * k1 * s1 * s6/(1 + (s6/KI)**q) - k2 * s2 * (N - s5) - k6 * s2 * s5
    ds3 = k2 * s2 * (N - s5) - k3 * s3 * (A - s6)
    ds4 = k3 * s3 * (A - s6) - k4 * s4 * s5 - kappa * (s4 - s7)
    ds5 = k2 * s2 * (N - s5) - k4 * s4 * s5 - k6 * s2 * s5
    ds6 = -2 * k1 * s1 * s6/(1 + (s6/KI)**q) + 2 * k3 * s3 * (A - s6) - k5 * s6
    ds7 = phi * kappa * (s4 - s7) - k * s7
    
    return np.array([ds1, ds2, ds3, ds4, ds5, ds6, ds7])


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
    
    # Define the grid
    start_time = args.minTime # default 0
    end_time = args.maxTime # default 10
    step = args.stepSize # default 0.01
    time_points = np.arange(start_time, end_time, step)
    
    params = {'Jin': 2.5, 'KI':0.52,
         'k1': 100., 'k2': 6., 'k3': 16., 'k4': 100., 'k5': 1.28, 'k6': 12.,
          'kappa': 13., 'q': 4, 'k': 1.8}

    #testing to see if it works
    #f(np.zeros(7), params)
    
    # initial conditions are taken from
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119821#pone-0119821-t002
    x_init = np.zeros(7)
    x_init[0] = np.random.uniform(0.15, 1.6, size=1)
    x_init[1] = np.random.uniform(0.19, 2.16, size = 1)
    x_init[2] = np.random.uniform(0.04, 0.2, size=1)
    x_init[3] = np.random.uniform(0.1, 0.35, size=1)
    x_init[4] = np.random.uniform(0.08, 0.3, size = 1)
    x_init[5] = np.random.uniform(0.14, 2.67, size = 1)
    x_init[6] = np.random.uniform(0.05, 0.1, size=1)
    
    # solve ode
    sol = odeint(lambda x, t: f(x, params), x_init, time_points)
    
    # create training data consisting of noisy measurements
    sampling_rate = 1
    noise = 0.0
    gly_data = sol[0::sampling_rate, :]
    gly_data += noise * gly_data.std(0) * np.random.randn(*gly_data.shape)

    gly_data = np.reshape(gly_data, (1,gly_data.shape[0], gly_data.shape[1]))
    
    hidden_layer_units = 256 # number of units for the hidden layer
    M = args.M # number of steps
    scheme = args.scheme # LMM scheme
    model = lmmNet(step, gly_data, M, scheme, hidden_layer_units)

    N_Iter = 10000
    model.train(N_Iter)
    
    gly_pred = odeint(ml_f, x_init, time_points, args=(model,))
    
    glycolysis = {}
    
    glycolysis['data'] = gly_data
    glycolysis['pred'] = gly_pred
    glycolysis['t'] = time_points
    
    # save to file
    with open('7-d_glycolysis.pkl', 'wb') as file:
            pickle.dump(glycolysis, file)