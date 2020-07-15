def f_NovakTyson_stochastic(x,t, noise):
    """
    ODEs for Novak-Tyson cell cycle model.
    There are two main classes of equations:
    1. synthesis/degradation of cyclin
    2. phosporylation/dephosporylation
    
    Args:
    x -- array of concentrations for the 7 biochemical species
    t -- time, ignored
    
    Return: a list of derivatives
    """
    
    cyclin = x[0]
    MPF = x[1]
    preMPF = x[2]
    cdc25P = x[3]
    wee1P = x[4]
    IEP = x[5]
    APC = x[6]
        
    CDK = CDK_total - MPF - preMPF
    
    # cyclin is degraded by APC that is off and on -- overall a weighted combination of the two rates
    k2 = v2_1 * (APC_total - APC) + v2_2 * APC
    
    k25 = v25_1*(cdc25_total - cdc25P) + v25_2*cdc25P
    kwee = vwee_1*wee1P + vwee_2*(wee1_total - wee1P)
    
    # synthesis and degradation of cyclin
    # k1 -- synthesis
    # k2 -- degradation
    # k3 -- dimer formation
    d_cyclin = k1 - k3 * cyclin * CDK - k2 * cyclin
    
    # phosphorylation and dephosphorylation of the CDK subunit via cdc25 (mutual activation) and wee1 (mutual inhibition)
    d_MPF = k3*cyclin*CDK - k2*MPF - kwee*MPF + k25*preMPF
    d_preMPF = -k2*preMPF + kwee*MPF - k25*preMPF
    d_cdc25P = ka*MPF*(cdc25_total - cdc25P)/(Ka + cdc25_total - cdc25P) - kb*PPase*cdc25P/(Kb + cdc25P)
    d_wee1P = ke*MPF*(wee1_total - wee1P)/(Ke + wee1_total - wee1P) - kf*PPase*wee1P/(Kf + wee1P)
    
    d_IEP = kg*MPF*(IE_total - IEP)/(Kg + IE_total - IEP) - kh*PPase*IEP/(Kh + IEP)
    d_APC = kc*IEP*(APC_total - APC)/(Kc + APC_total - APC) - kd*PPase*APC/(Kd + APC)
    
    derivatives = np.array([d_cyclin, d_MPF, d_preMPF, d_cdc25P, d_wee1P, d_IEP, d_APC]) 
    derivatives += noise * np.random.randn(derivatives.shape[0])
    
    return derivatives


def sample_dynamics(start_time, end_time, step_size, f, x0, integrator='scipy', noise_strength=0):
    """
    Create tensor array for training by solving the initial value problem and adding noise.
    The ODE is integrated using LSODA from scipy.
    
    Args:
        noise_strength
        start_time
        end_time
        step_size
        f -- the function to integrate
        x0 -- the initial conditions
        integrator -- the numerical method library to use (currently supports only scipy)
        
    Returns:
        A tuple consisting of
        * time points of the grid
        * a tensor array with shape 1 x -1 as expected by LmmNet function call
    """
    time_points = np.arange(start_time, end_time, step_size)
    
    # choice of bips integrator (this is future work)
    if integrator == 'scipy':
        array = odeint(f, x0, time_points)
    elif integrator == 'bips':
        array = integrate_bips(f, x0, time_points)
        
    array += noise_strength * array.std(0) * np.random.randn(array.shape[0], array.shape[1])
    training_data = np.reshape(array, (1,array.shape[0], array.shape[1]))
    
    return time_points, tf.convert_to_tensor(training_data, dtype=tf.float32)
