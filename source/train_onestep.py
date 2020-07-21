import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def train_classic(data,model,plot=False,model_type=None, figure_path='./plots'):
    
    """
    Train the input data {X, y}.
    
    Arguments:
    
    data -- multi-index dataframe of time-series measurements, preprocessed by interpolating and filtering
    model -- a selected machine learning model
    plot -- decide to plot the result or not
    model_type -- determine the input model
    
    Returns:
    model_dict -- a trained model dictionary for each target
    score_dict -- a training score dictionary or each target
    
    """
            
    model_dict = {}
    score_dict = {}

    avg_score = 0
    n = 0

    for target_idx in data.columns:
    
        # All we want to train are targets
        if target_idx[0] == 'feature':
            continue
        target = target_idx[1]
        
        # TODO: create the data matrix X and the target vector y
        X = data['feature'].values.tolist() # YOUR CODE HERE
        y = data[target_idx].values.tolist() # YOUR CODE HERE
        
        if model_type == 'tpot':
            X = np.array(X)
            y = np.array(y)
        
        # TODO: train the model
        # IMPORTANT: clone the model to train a different one for each target
        if model_type == 'tpot':
            # if TPOT, use the best pipeline found
            model_dict[target] = clone(model).fit(X,y).fitted_pipeline_ # YOUR CODE HERE
        else:
            # if RF/NN/LR, simply fit X and y
            model_dict[target] = clone(model).fit(X,y) # YOUR CODE HERE
        
        # Plot results, if required
        if plot:
            
            # The training_plot function is defined below for you to complete
            CV_plot = training_plot(model_dict[target],
                                    target,X,y,
                                    cv=ShuffleSplit())
            
            axis = plt.gca()
            axis.set_ylim([-0.1, 1.1])
            
            strip_target = ''.join([char for char in target if char != '/'])
            print(strip_target)
            
            CV_plot.savefig(figure_path + strip_target + '_' + model_type + '_CV_plot.pdf',transparent=False)
            
            plt.show()
    
        # evaluate the model score
        # Every model in sklearn API has its own default scoring metric (see the respective docs), but can be easily accesed via the score method
        score = model_dict[target].score(X,y) ## YOUR CODE HERE
            
        print('Target: {}, CV Pearson R2 coefficient: {:f}'.format(target,score))
        score_dict[target] = score
    
    # TODO: compute the average score over all targets
    avg_score = sum(score_dict.values())/len(score_dict.values()) #YOUR CODE HERE
    print('Average training score:', avg_score)
    
    return model_dict,score_dict



def training_plot(estimator,title,X,y,
                  cv=None,n_jobs=1, 
                  train_sizes=np.linspace(.1, 1.0, 5)):
    
    """
    Generate a plot in training process.

    Arguements:
    
    estimator -- a machine learning model.
    title -- a title for the chart.
    X -- array of features.
    y -- target array corresponded to X.
    cv -- a cross-validation generator.
    n_jobs : a number of jobs to run in parallel.
    
    Return:
    plt -- a desired plot.
    
    """
    
    plt.figure()
    plt.title(title)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    # call the learning_curve function to get scores for different training sizes
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs) # YOUR CODE HERE
    
    # compute the mean and standard deviation of the scores
    train_scores_mean = np.mean(train_scores, axis=1) #YOUR CODE HERE
    train_scores_std = np.std(train_scores, axis=1) #YOUR CODE HERE
    
    test_scores_mean = np.mean(test_scores, axis=1) #YOUR CODE HERE
    test_scores_std = np.std(test_scores, axis=1) #YOUR CODE HERE
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def generate_dataset(data, strain_list, feature_list, target_list, n_dim):
    
    """
    Generate and augment the training data {X, y} for model fitting, using savgol filter as the smoothing method.
    
    Arguments:
    
    data -- time-series data frame of measurements, with 'Strain' as the index
    strain_list -- list of unique strains in `data`
    feature_list -- list of features to be used
    target_list -- list of targets
    n_dim -- number of data points to generate via interpolation
    
    Returns:
    ml_data -- a pandas multi-index dataframe containing features x and targets y.
    
    """
    
    ml_data = pd.DataFrame()
    
    for strain in strain_list:
        measurement_data = {}

        # Interpolate -> Filter -> Add to the table
        for measurement in feature_list + target_list:

            # extract measurement for the specific strain
            measurement_series = data.loc[strain][measurement]
            T = data.loc[strain]['Time'] # series of time points
            
            ## extract the start time and end time and the time step
            minT,maxT = min(T),max(T) # start time and end time
            delT = (maxT - minT)/n_dim # time step for interpolation
        
            # Interpolate data
            interpolation = interp1d(T,
                                     measurement_series.tolist(),
                                     kind='linear')
            
            # generate time points to interpolate over using np.linspace
            time_points = np.linspace(minT,maxT,n_dim)
            
            # Consider the interpolated data over time
            interpolated_measurement = interpolation(time_points)
            
            # apply savgol filter to interpolated measurement, using window length of 7 and polyorder of 2
            filtered_measurement = savgol_filter(interpolated_measurement,
                                                 window_length=7,
                                                 polyorder=2)

            # fill in the data to a multi-index data frame
            if measurement in feature_list:
                # use the filtered measurement of this enzyme as features
                measurement_data[('feature',measurement)] = filtered_measurement # YOUR CODE HERE
            if measurement in target_list:
                # use the filtered measurment of this metabolite as a feature
                measurement_data[('feature',measurement)] = filtered_measurement # YOUR CODE HERE
                # additionally compute gradients of the filtered measurement and use it as target
                measurement_data[('target',measurement)] = np.gradient([point/delT for point in filtered_measurement])
   
        # Create a table
        strain_data = pd.DataFrame(measurement_data,
                                   index=pd.MultiIndex.from_product([[strain],np.linspace(minT,maxT,n_dim)],
                                   names=['Strain', 'Time']))
        ml_data = pd.concat([ml_data,strain_data])
        
    return ml_data
