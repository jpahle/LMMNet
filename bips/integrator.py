import scipy.integrate as integrate
from scipy.integrate import ode
import ipywidgets as widgets
import numpy as np

def integrate_bips(g,y0,times):
    
    """
    A custom integrator by BIPS for solving initial value problem, suitable for a function learned by machine learning.
    This makes sure that the concentrations do not leave space of positive values.
    
    Arguments:
    g (function) -- the dynamics to be integrated with x as first argument and t as second argument
    y0 (list) -- an initial condition as a list of concentrations
    times (list) -- a list of time coordinate
    
    Return:
    x (list) -- a solution of the ode problem
    
    """
    
    # Set up ODE
    f = lambda t,x: g(x,t)
    r = ode(f).set_integrator('dopri5',
                              nsteps=1e4,
                              atol=1e-3)

    r.set_initial_value(y0,times[0])

    widgets.FloatProgress(min=0, max=max(times))

    # Preform integration
    x = [y0,]
    currentT = times[0]
    max_delT = 10

    for nextT in times[1:]:

        while r.t < nextT:

            if nextT-currentT < max_delT:
                dt = nextT-currentT
            else:
                dt = max_delT

            value = r.integrate(r.t + dt)
            #print("value: ", value)
            value = np.max([value, np.zeros(2)], axis=0)
            currentT = r.t
            
            f.value = currentT
            r.set_initial_value(value,currentT)

        x.append(value)
    return np.array(x)