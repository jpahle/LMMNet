import pandas as pd
import numpy as np
import train_onestep
import predict_onestep
import predict_lmmNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

def create_data(data_tensor, times, cols, num=0):
    """
    A helper function to create dataframe from tensors
    
    Args:
    data_tensor -- the data tensor of 3 dimensions (trajectories, time points, species)
    num -- the 'strain' to be used as index value (row names)
    cols -- the column names for the dataframe
    times -- time points
    """
    df = pd.DataFrame(data_tensor.numpy()[0])
    df.columns = cols
    df['Strain'] = [num] * df.shape[0]
    df = df.set_index('Strain')
    df['Time'] = times
    print('Shape of the dataframe is:', df.shape)
    return df


def create_data_numpy(data_numpy, times, cols, num=0):
    """
    A helper function to create dataframe from tensors
    
    Args:
    data_tensor -- the data tensor of 3 dimensions (trajectories, time points, species)
    num -- the 'strain' to be used as index value (row names)
    cols -- the column names for the dataframe
    times -- time points
    """
    df = pd.DataFrame(data_numpy)
    df.columns = cols
    df['Strain'] = [num] * df.shape[0]
    df = df.set_index('Strain')
    df['Time'] = times
    print('Shape of the dataframe is:', df.shape)
    return df


# train model and make predictions
def end_to_end_training(df, df_train, df_test, feature_list, target_list, plot_size):
    """
    Do end-to-end training with random forest
    df_train: training data augmented
    df_test: test data raw
    df: training data raw
    """
    rf_model = RandomForestRegressor(n_estimators=20)
    figure_path = './plots/'
    
    rf_dict, score_dict = train_onestep.train_classic(df_train, rf_model, plot=True,model_type='random_forest', figure_path=figure_path)
    
    train_onestep.compute_error(df_train,rf_dict,plot=True,model_type='random_forest')
    
    # now we make predictions via numerical integration
    # note that in predict_integrate, the function expects a normal dataframe and not the time-series multi-index dataframe
    time_points, predictions = predict_onestep.predict_integrate(df_test, df, rf_dict, target_list, feature_list, title='damped harmonic', plot=True,model_type='random_forest', subplots=plot_size, bio=False)
    
    return time_points, predictions