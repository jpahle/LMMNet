import numpy as np


def ruoff(x, t, params=None, N = 1., A = 4., phi = 0.1):
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
    
    if params == None:
        params = {'Jin': 2.5, 'KI':0.52,
         'k1': 100., 'k2': 6., 'k3': 16., 'k4': 100., 'k5': 1.28, 'k6': 12.,
          'kappa': 13., 'q': 4, 'k': 1.8}
        
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
