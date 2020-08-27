import numpy as np

def normalize(v):
    """
    Return the normalized array of v
    """
    
    return v/np.linalg.norm(v)