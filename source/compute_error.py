# function to compute RMSE error for every metabolite and the total RMSE
# make sure that the directory specified in figure_path exists! otherwise the execution will return an error.
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compute_error(data,model_dict,plot=False, model_type = None, figure_path = './plots/'):
    
    """
    To check the error of predicted derivative.
    
    Arguments:
    
    data -- time-series data of measurements, preprocessed by interpolating and filtering
    model_dict -- a dictionary of trained models or each target
    plot -- decide to plot the result or not
    model_type -- type of model used
    """
    
    # list of errors
    # this contains the error for every metabolite (target)
    error_list = []

    for target in model_dict:
        # Extract input target
        y_test = data[('target',target)].values
    
        # Extract predicted target
        feature_list = [('feature',feature) for feature in data['feature'].columns]
        target_data = data[feature_list]
        y_prediction = model_dict[target].predict(target_data.values)
    
        # TODO: Compute squared error and append it to the list of errors
        ## YOUR CODE HERE
        error = [y_p - y_t for y_p,y_t in zip(y_prediction,y_test)]
        error_list.append(error)
        
        # Compute mean and standard deviation of squared error
        ## YOUR CODE HERE
        mu = np.mean(error)
        sigma = np.std(error)
        print(target,'RMSE:',mu,'standard deviation:',sigma)
        
        if plot:
            plt.figure(figsize=(13,4))
            plt.subplot(121)
            sns.distplot(error)
            
            plt.title(target + ' Derivative '+ 'Error Residual Histogram')
            plt.xlabel('Derivative Residual Error')
            plt.ylabel('Probability Density')
    
            plt.subplot(122)
            error_plot(target,y_prediction,y_test) # this function is provided below
    
            strip_target = ''.join([char for char in target if char != '/'])
            plt.savefig(figure_path + strip_target +'_'+ model_type + '_Error_Residuals.pdf')
            plt.show()

    # TODO: compute total error from the error list
    ## YOUR CODE HERE
    error_magnitude = [0]*len(error_list[0])

    for error in error_list:
        error_magnitude = [em + e**2 for em,e in zip(error_magnitude,error)]
        error_magnitude = [math.sqrt(e) for e in error_magnitude]

    mu = np.mean(error_magnitude)
    sigma = np.std(error_magnitude)
    print('Total Derivative','Mean Error:',mu,'Error Standard Deviation:',sigma)
    
    if plot:
        sns.distplot(error_magnitude)
        plt.title('Total Derivative Error Histogram')
        plt.show()
        
        

def error_plot(name,pred,real):
    
    """
    Generate a plot from detecting error of derivatives.

    Arguements:
    
    name -- a name for the title.
    pred -- a list of predicted derivatives
    real -- a list of actual derivatives
    
    """

    plt.scatter(pred,real)
    plt.title(name + ' Predicted vs. Actual')
    
    axis = plt.gca()
    axis.plot([-120,120], [-120,120], ls="--", c=".3")
    
    padding_y = (max(real) - min(real))*0.1
    plt.ylim(min(real)-padding_y,max(real)+padding_y)
    
    padding_x = (max(pred) - min(pred))*0.1
    plt.xlim(min(pred)-padding_x,max(pred)+padding_x)
    
    plt.xlabel('Predicted ' + name)
    plt.ylabel('Actual ' + name)
 