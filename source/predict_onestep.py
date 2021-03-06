from scipy.integrate import ode
import random
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math

# integrate
def int_ode(g,y0,times,solver='scipy'):
    
    """
    Integration function corresponded to the ode, generated by ml_ode.
    
    Arguments:
    f -- an ode equation to be integrated
    y0 -- an initial condition as a list of concentrations
    times -- a list of time coordinate
    solver -- string of package used for ode solver
    
    Return:
    x -- a solution of the ode problem
    
    """
    
    if solver == 'assimulo':
        from assimulo.problem import Explicit_Problem
        from assimulo.solvers import Dopri5
        
        # Set up ODE
        rhs = lambda t,x: g(x,t)
        model = Explicit_Problem(rhs,y0,min(times))
        sim = Dopri5(model)
        
        # Preform integration
        _,x = sim.simulate(max(times),max(times))
        return np.array(x)[np.array(times).astype(int)].tolist()
    
    elif solver == 'scipy':
        # Set up ODE
        f = lambda t,x: g(x,t)
        r = ode(f).set_integrator('dopri5',
                                  nsteps=1e4,
                                  atol=1e-5)
    
        r.set_initial_value(y0,times[0])
    
        #widgets.FloatProgress(min=0, max=max(times))
    
        # Preform integration
        x = [y0,]
        currentT = times[0]
        max_delT = 10
    
        for nextT in times[1:]:
        
            while r.t < nextT:
            
                if nextT-currentT < max_delT:
                    dt = nextT-currentT
                else:
                    dt = max_delT
                
                value = r.integrate(r.t + dt)
                currentT = r.t

                f.value = currentT
            
            x.append(value)
        return x
    
    
    
# define the derivatives
def ml_ode(model_dict, data, targets, features, time_index='Hour'):
    
    """
    Set up an ODE.
    
    Arguments:
    model_type -- a string for desired model
    data -- raw time-series data of measurements
    targets -- list of targets
    features -- list of features
    time_index -- a string labelel for time index of the input data
    
    Return:
    f - an output ODE
    
    """
    
    # Create interpolations for each feature
    ml_interpolation = {}
    
    for feature in data.columns:
        feature_columns = feature
        
        if isinstance(feature,tuple):
            if feature[0]=='feature':
                feature = feature[1]
            else:
                continue

        if feature in features:    
            X,y = data.reset_index()[time_index].tolist(), data[feature_columns].tolist()

            ml_interpolation[feature] = interp1d(X,y)
            
    # Define the function to be integrated
    def f(x,t):
        x_dot = []
        
        # Create derivatives for each target
        for target in targets:
            x_pred = []
            
            # loop over all species
            for feature in data.columns:
                if isinstance(feature,tuple):
                    if feature[0]=='feature':
                        feature = feature[1]
                    else:
                        continue
                
                if feature in features:
                    x_pred = np.append(x_pred, ml_interpolation[feature](t))
                elif feature in targets:
                    x_pred = np.append(x_pred, x[targets.index(feature)])
                
            model_prediction = model_dict[target].predict(x_pred.reshape(1,-1))
            x_dot = np.append(x_dot,model_prediction)   
            
        return x_dot
    return f


# write a function to integrate the dynamics and predict time points
def predict_integrate(ts_data,tr_data,model_dict,targets,features,title,
              plot=False,model_type=None,solver='scipy', figure_path = './plots/', subplots = (3,2), bio=True):
    
    """
    Integrate the learned 'ODE' and use it for simulations
    
    Arguments:
    
    tr_data -- raw dataframe of measurements used for training
    ts_data -- raw dataframe of measurements used for testing
    model_dict -- a dictionary of trained models or each target
    title -- a title for the plot
    targets -- list of targets
    features -- list of features
    plot -- decide to plot the result or not
    model_type -- determine the input model
    solver -- string of package used for ode solver
    bio -- biochemical species? To restrict to positive concentrations
    
    """
    
    rmse_average = []
    rmse_percent = []
    
    ts = ts_data 
    
    # Get a randomed strain
    strains = ts.index.get_level_values(0).unique().tolist()
    strain = random.sample(strains,1)
    
    test_data = ts.loc[strain]
        
    # TODO: get the initial conditions from test_data
    y0 = test_data[targets].iloc[0].tolist()
    
    # TODO: call ml_ode function to construct the 'ODE'
    g = ml_ode(model_dict, 
               test_data, 
               targets, 
               features, 
               time_index='Time')

    # Get the time points
    times = test_data.reset_index()['Time'].tolist()
        
    # TODO: call int_ode to integrate the 'ODE' g
    fit = int_ode(g,y0,times,solver=solver)
        
    # Format the output as a table
    fit_data = pd.DataFrame(fit, 
                            index=times, 
                            columns = targets).rename_axis('Time')
    
    # Set up real data and predicted targets
    real = test_data[targets]
    pred = fit_data
        
    # Display them
    print('Real data:')
    display(real)
    print('Predicted data:')
    display(pred)
        
        
    for metabolite in fit_data.columns:
        t,X = times, real[metabolite].tolist()
        real_fcn = interp1d(t,X)
        pred_fcn = interp1d(times,pred[metabolite])
            
        '''
        Optional 
        times =  real[metabolite].dropna().index.tolist()
        real_fcn = interp1d(times,real[metabolite].dropna())
        pred_fcn = interp1d(times,pred[metabolite].loc[times])
        '''
            
        # Calculate RMSE average
        integrand = lambda t: (real_fcn(t) - pred_fcn(t))**2
        rmse = math.sqrt(quad(integrand,min(times),max(times),limit=200)[0])
        rmse_average.append(rmse)
            
         # Calculate RMSE percentage
        percent_integrand = lambda t: abs(real_fcn(t) - pred_fcn(t))/abs(real_fcn(t)*max(times))
        rmsep = math.sqrt(quad(percent_integrand,min(times),max(times),limit=200)[0])
        rmse_percent.append(rmsep)
        
        print('ML Fit:',metabolite,rmse,
              'RMSE percentage:',rmsep*100)
    
    print('ML model aggregate error')
    print('Average RMSE:',sum(rmse_average)/len(rmse_average))
    print('Total percentage error:',sum(rmse_percent)/len(rmse_percent)*100)
        
    if plot:
        tr = tr_data
        tr_strains = tr_data.index.get_level_values(0).unique().tolist()
        fitT = list(map(list, zip(*fit)))
        
        # Create interpolation functions for each feature
        interp_f = {}
            
        for feature in test_data.columns:
            t,X = test_data.reset_index()['Time'].tolist(), test_data[feature].tolist()
            interp_f[feature] = interp1d(t,X)
        
        plt.figure(figsize=(12,8))
        
        common_targets = targets
        for i,target in enumerate(common_targets):
            plt.subplot(subplots[0],subplots[1],i+1)
            
            for strain in tr_strains:
                strain_interp_f = {}
                strain_df = tr.loc[strain]
                
                X,y = strain_df.reset_index()['Time'].tolist(), strain_df[target].tolist()
                strain_interp_f[target] = interp1d(X,y)
                
                actual_data = [strain_interp_f[target](t) for t in times]
                
                train_line, = plt.plot(times,actual_data,'g--')
                    
            actual_data = [interp_f[target](t) for t in times]
            
            if bio:
                pos_pred = [max(fitT[i][j],0) for j,t in enumerate(times)]
            else:
                pos_pred = [fitT[i][j] for j,t in enumerate(times)]
            prediction_line, = plt.plot(times,pos_pred)
            
            test_line, = plt.plot(times,actual_data,'r--')
            
            plt.ylabel(target)
            plt.xlabel('Time [h]')
            #plt.xlim([0,72])
    
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.subplots_adjust(bottom=0.12)
        plt.suptitle('Prediction of ' + title + ' Strain Dynamics', fontsize=18)
        plt.figlegend((train_line,test_line,prediction_line), 
                      ('Training Set Data','Test Data','Machine Learning Model Prediction'), 
                      loc = 'lower center', ncol=5, labelspacing=0. ) 
            
        plt.savefig(figure_path + title + model_type +'_prediction.eps', format='eps', dpi=600)
        plt.show()
        
    return times, pred