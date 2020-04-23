import numpy as np

def add_noise(lorenz_data, noise_strength):
    """
    Add noise to the training data
    
    Args:
        lorenz_data -- the dataset to use
        noise_strength
        
    Returns:
        a dataset with shape 1 x -1 as expected by LmmNet function call
    """
    # add Gaussian noise scaled by standard deviation, for every one of the three dimensions
    lorenz_data += noise_strength * lorenz_data.std(0) * np.random.randn(lorenz_data.shape[0], lorenz_data.shape[1])

    lorenz_data = np.reshape(lorenz_data, (1,lorenz_data.shape[0], lorenz_data.shape[1]))
    
    return lorenz_data


def ml_f(x, t, model):
    """
    Define the derivatives learned by ML
    I think this is the best implementation, more robust than flatten()
    
    Args:
    x -- values for the current time point
    t -- time, dummy argument to conform with scipy's API
    model -- the learned ML model
    """
    return np.ravel(model.predict(x.reshape(1,-1)))