import numpy as np
from .train import create_training_data

def f_NovakTyson(x,t):
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

    return [d_cyclin, d_MPF, d_preMPF, d_cdc25P, d_wee1P, d_IEP, d_APC]

def simulate_default(debug=False):
    """
    Simulate the Novak Tyson Cell Cycle Model
    
    Returns:
    - time points
    - virtual time-series measurements for each biochemical species
    """
    
    # define the simulation time
    tfirst, tlast = 0, 1500
    step_size = 0.2

    # define initial conditions
    cyclin = 0
    MPF = 0
    preMPF = 0
    cdc25P = 0
    wee1P = 0
    IEP = 1
    APC = 1
    x0 = np.array([cyclin,MPF,preMPF,cdc25P,wee1P,IEP,APC])
    
    # define default parameters
    default = {'k1':1, 'k3':0.005,
              'ka':.02,'Ka':.1,'kb':.1,'Kb':1, 'kc':.13, 'Kc':.01, 'kd':.13, 'Kd':1,
             'v2_1':.005, 'v2_2':.25, 'vwee_1':.01, 'vwee_2':1, 'v25_1':0.5*.017, 'v25_2':0.5*.17,
             'ke':.02, 'Ke':1, 'kf':.1, 'Kf':1, 'kg':.02, 'Kg':.01, 'kh':.15, 'Kh':.01,
             'wee1_total':1, 'PPase':1, 'CDK_total':100, 'cdc25_total':5,'IE_total':1, 'APC_total':1}
    for key,val in default.items():
        globals()[key]=val
        
    if debug:
        return f_NovakTyson(x0, 0)
    
    time_points, novak_data = create_training_data(tfirst, tlast, step_size, f_NovakTyson, x0)
    
    return time_points, novak_data

def simulate_custom(tfirst=0, tlast=300, step_size=0.2, cyclin=22, MPF=11):
    """
    Simulate the Novak Tyson Cell Cycle Model
    
    Returns:
    - time points
    - virtual time-series measurements for each biochemical species
    """
    

    # define initial conditions
    preMPF = 0
    cdc25P = 0
    wee1P = 0
    IEP = 1
    APC = 1
    x0 = np.array([cyclin,MPF,preMPF,cdc25P,wee1P,IEP,APC])
    
    # define default parameters
    default = {'k1':1, 'k3':0.005,
              'ka':.02,'Ka':.1,'kb':.1,'Kb':1, 'kc':.13, 'Kc':.01, 'kd':.13, 'Kd':1,
             'v2_1':.005, 'v2_2':.25, 'vwee_1':.01, 'vwee_2':1, 'v25_1':0.5*.017, 'v25_2':0.5*.17,
             'ke':.02, 'Ke':1, 'kf':.1, 'Kf':1, 'kg':.02, 'Kg':.01, 'kh':.15, 'Kh':.01,
             'wee1_total':1, 'PPase':1, 'CDK_total':100, 'cdc25_total':5,'IE_total':1, 'APC_total':1}
    for key,val in default.items():
        globals()[key]=val
    
    time_points, novak_data = create_training_data(tfirst, tlast, step_size, f_NovakTyson, x0)
    
    return time_points, novak_data