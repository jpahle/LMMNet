import numpy as np

def normalize(v):
    """
    Return the normalized array of v
    """
    
    return v/np.linalg.norm(v)

def get_mse(pred, dat):
    e1 = predict_lmmNet.compute_MSE(pred, dat, 0)
    e2 = predict_lmmNet.compute_MSE(pred, dat, 1)
    e3 = predict_lmmNet.compute_MSE(pred, dat, 2)
    e4 = predict_lmmNet.compute_MSE(pred, dat, 3)
    e5 = predict_lmmNet.compute_MSE(pred, dat, 4)
    e6 = predict_lmmNet.compute_MSE(pred, dat, 5)
    e7 = predict_lmmNet.compute_MSE(pred, dat, 6)
    
    return (e1, e2, e3, e4, e5, e6, e7)