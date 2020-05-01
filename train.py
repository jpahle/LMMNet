# Kevin Siswandi
# April 2020

from bips.lmmNet import lmmNet
from bips.utils import *
from bips.systems import *
from bips.integrator import *

import numpy as np
from scipy.integrate import odeint
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate a dynamical system and reconstruct the dynamics with lmmNet.')
parser.add_argument('--noise', action='store',default=0.0, type=float,
                   help='strenght of noise to be added to the training data (default: 0.00)')
parser.add_argument('--system', action='store',default='Cubic', type=str,
                   help='Choose a system to simulate and discover.')
parser.add_argument('--filename', action='store', type=str,
                   help='The name of the file to save the output to.')
parser.add_argument('--integrator', action='store', type=str, default='scipy',
                   help='Integrator to use: either scipy or bips')
parser.add_argument('--M', type=int, default=1,
                   help='the number of steps to use.')

if __name__ == "__main__":
    
    global args
    args = parser.parse_args()
    
    # EDIT THIS FOR YOUR PURPOSE
    # DEFINE THE PROBLEM
    
    if args.system == 'Bier':
        # 2-D Bier settings
        t0, T, h = 0, 500, 0.01 #seconds
        x0 = np.array([4, 3]) #initial conditions: ATP = 4, G = 3 -- default Bier model
        f = bier
    elif args.system == 'Cubic':
        # 2-D Cubic settings
        t0, T, h = 0, 25, 0.05
        x0 = np.array([2,0]) # initial conditions -- default cubic problem
        f = cubic # the system to study
    elif args.system == 'Lorenz':
        # 3-D Lorenz settings
        x0 = np.array([-8.0, 7.0, 27])
        t0, T, h = 0, 25, 0.01
        f = lorenz
    elif args.system == 'Ruoff':
        # 7-D Glycolysis settings
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119821#pone-0119821-t002
        x0 = np.zeros(7)
        x0[0] = np.random.uniform(0.15, 1.6, size=1)
        x0[1] = np.random.uniform(0.19, 2.16, size = 1)
        x0[2] = np.random.uniform(0.04, 0.2, size=1)
        x0[3] = np.random.uniform(0.1, 0.35, size=1)
        x0[4] = np.random.uniform(0.08, 0.3, size = 1)
        x0[5] = np.random.uniform(0.14, 2.67, size = 1)
        x0[6] = np.random.uniform(0.05, 0.1, size=1)     
        t0, T, h = 0, 10, 0.01
        f = ruoff
        
    hidden_layer_units = 256 # number of units for the hidden layer
    M = 1 # number of steps
    scheme = 'AM' # LMM scheme
    
    ####
    #### END EDITABLE PARAMETERS

    
    time_points = np.arange(t0, T, h)
    
    if args.integrator == 'scipy':
        data = odeint(f, x0, time_points)
    elif args.integrator == 'bips':
        data = integrate_bips(f, x0, time_points)
    
    noise_strength = args.noise
    data = add_noise(data, noise_strength)

    model = lmmNet(h, data, M, scheme, hidden_layer_units)

    N_Iter = 10000
    model.train(N_Iter)
    
    # TODO: change to a new integrator
    if args.integrator == 'scipy':
        pred = odeint(ml_f, x0, time_points, args=(model,))
    elif args.integrator == 'bips':
        pred = integrate_bips(lambda x,t: ml_f(x,t, model), x0, time_points)
    
    result_dict = {}
    result_dict['data'] = data
    result_dict['pred'] = pred
    result_dict['t'] = time_points

    # save to file
    with open(str(args.filename), 'wb') as file:
            pickle.dump(result_dict, file)