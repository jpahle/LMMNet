import numpy as np
from scipy.integrate import odeint
import train_lmmNet

def f_bier(x,t, params=None):
    """
    2-D Yeast Glycolytic oscillator model
    
    Args:
        x -- a 2 x 1 vector of measurements
        t -- time, ignored
        
    Return:
        A numpy array containing the derivatives
    """
    if params == None:
        # default parameter values
        Vin = 0.36
        k1 = 0.02
        kp = 6
        km = 12
    else:
        Vin = params['Vin']
        k1 = params['k1']
        kp = params['kp']
        km = params['km']
    
    r1 = 2 * k1 * x[0] * x[1] - kp * x[0]/(x[0] + km) # ATP
    r2 = Vin - k1 * x[0] * x[1] #G
    
    return np.ravel(np.array([r1, r2]))

def simulate_default():
    
    # generate data
    t0 = 0
    T = 1000 # in seconds
    h = 0.2
    time_points = np.arange(t0, T, h)
    x0 = np.array([4, 3]) #initial conditions: ATP = 4, G = 3
    
    bier_data = odeint(f_bier, x0, time_points)
    
    return time_points, bier_data

        
def simulate_custom(t0 = 0, T = 1000, h=0.2, params={'Vin': 0.36, 'k1': 0.02, 'kp':4, 'km':15}, x0 = [4,3]):
    """
    The default parameters in the params argument are for damped oscillation
    """
    
    x0 = np.array(x0) #initial conditions: ATP = 4, G = 3 -- default Bier model
    time_points = np.arange(t0, T, h)
    bier_data = odeint(lambda x, t: f_bier(x, t, params), x0, time_points)
    
    return time_points, bier_data