from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np

def generateTSDataSet(dataframe,features,targets,n_points=100):
    """
    Modified by Kevin for bioLearningPractical
    
    Generate a smooth time-series data, interpolated from experimental measurements, and
    compute the time derivatives associated with the target variables.
    
    Args:
    - n_points = number of data points to interpolate
    - dataframe = the raw data (e.g. from experiments)
    - features = set of feature columns (metabolite + protein concentrations)
    - targets = set of target columns for which the time derivative is to be computed
    
    Return: a multi-index Pandas dataframe
    """
    strains = tuple(dataframe.index.get_level_values(0).unique())
    numSamples = len(strains)
    print( 'Total Time Series in Data Set: ', numSamples )
    
    ml_df = pd.DataFrame()
    for strain in strains:
        strain_series = {}
        strain_df = dataframe.loc[(strain,slice(None)),:]
        strain_df.index = strain_df.index.get_level_values(1)
        
        #Interpolate & Smooth Each Feature & Target Then Add To Series
        for measurement in features + targets:
            #Extract Measurement
            measurement_series = strain_df[measurement].dropna()
            
            #Generate n_points interpolated points
            times = measurement_series.index.tolist()
            deltaT = (max(times) - min(times))/n_points
            
            measurement_fun = interp1d(times,
                                       measurement_series.tolist(),kind='linear')
            interpolated_measurement = measurement_fun(np.linspace(min(times),max(times),n_points))
            
            # smooth and interpolate
            smoothed_measurement = savgol_filter(interpolated_measurement,7,2)
            
            #the interpolated points can be used directly as features
            if measurement in features:
                #added ReLU
                strain_series[('feature',measurement)]=np.maximum(smoothed_measurement, 0.000000001)
            
            #in addition compute derivative for targets
            if measurement in targets:
                strain_series[('feature',measurement)]=np.maximum(smoothed_measurement, 0.000000001)
                strain_series[('target',measurement)]=np.gradient([point/deltaT for point in smoothed_measurement])
        
        strain_df = pd.DataFrame(strain_series,
                                 index=pd.MultiIndex.from_product([[strain],np.linspace(min(times),max(times),n_points)],
                                                             names=['Strain', 'Time (h)']))
        ml_df = pd.concat([ml_df,strain_df])

    return ml_df