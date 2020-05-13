from model import lmmNet
import numpy as np
import tensorflow as tf

def predict_fn(x, t, model):
    """
    Define the derivatives learned by ML
    I think this is the best implementation, more robust than flatten()
    
    Args:
    x -- values for the current time point
    t -- time, dummy argument to conform with scipy's API
    model -- the learned ML model
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return np.ravel(model.predict(x.reshape(1,-1)))