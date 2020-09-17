import numpy as np

def normalize(v):
    """
    Return the normalized array of v
    """
    
    return v/np.linalg.norm(v)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_mse(pred, dat):
    e1 = predict_lmmNet.compute_MSE(pred, dat, 0)
    e2 = predict_lmmNet.compute_MSE(pred, dat, 1)
    e3 = predict_lmmNet.compute_MSE(pred, dat, 2)
    e4 = predict_lmmNet.compute_MSE(pred, dat, 3)
    e5 = predict_lmmNet.compute_MSE(pred, dat, 4)
    e6 = predict_lmmNet.compute_MSE(pred, dat, 5)
    e7 = predict_lmmNet.compute_MSE(pred, dat, 6)
    
    return (e1, e2, e3, e4, e5, e6, e7)

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def get_dtw(pred, dat):
    e1 = fastdtw(pred[:,0], dat[:,0], dist=euclidean)[0]
    e2 = fastdtw(pred[:,1], dat[:,1], dist=euclidean)[0]
    e3 = fastdtw(pred[:,2], dat[:,2], dist=euclidean)[0]
    e4 = fastdtw(pred[:,3], dat[:,3], dist=euclidean)[0]
    e5 = fastdtw(pred[:,4], dat[:,4], dist=euclidean)[0]
    e6 = fastdtw(pred[:,5], dat[:,5], dist=euclidean)[0]
    e7 = fastdtw(pred[:,6], dat[:,6], dist=euclidean)[0]
    
    e1 /= np.linalg.norm(test_data[:,0], 2)**2
    e2 /= np.linalg.norm(test_data[:,0], 2)**2
    e3 /= np.linalg.norm(test_data[:,0], 2)**2
    e4 /= np.linalg.norm(test_data[:,0], 2)**2
    e5 /= np.linalg.norm(test_data[:,0], 2)**2
    e6 /= np.linalg.norm(test_data[:,0], 2)**2
    e7 /= np.linalg.norm(test_data[:,0], 2)**2
    return (e1, e2, e3, e4, e5, e6, e7)

from scipy.stats import wasserstein_distance

# compute Wasserstein distance
def get_was(pred, dat):    
    e1 = wasserstein_distance(pred[:,0], dat[:,0])
    e2 = wasserstein_distance(pred[:,1], dat[:,1])
    e3 = wasserstein_distance(pred[:,2], dat[:,2])
    e4 = wasserstein_distance(pred[:,3], dat[:,3])
    e5 = wasserstein_distance(pred[:,4], dat[:,4])
    e6 = wasserstein_distance(pred[:,5], dat[:,5])
    e7 = wasserstein_distance(pred[:,6], dat[:,6])
    
    return (e1, e2, e3, e4, e5, e6, e7)