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
            T = data.loc[strain]['Hour'] # series of time points
            
            ## TODO: extract the start time and end time and the time step
            minT,maxT = min(T),max(T) # start time and end time
            delT = (maxT - minT)/n_dim # time step for interpolation
        
            # Interpolate data
            interpolation = interp1d(T,
                                     measurement_series.tolist(),
                                     kind='linear')
            
            # TODO: generate time points to interpolate over using np.linspace
            time_points = np.linspace(minT,maxT,n_dim)
            
            # Consider the interpolated data over time
            interpolated_measurement = interpolation(time_points)
            
            # TODO: apply savgol filter to interpolated measurement, using window length of 7 and polyorder of 2
            filtered_measurement = savgol_filter(interpolated_measurement,
                                                 window_length=7,
                                                 polyorder=2)

            # TODO: fill in the data to a multi-index data frame
            if measurement in feature_list:
                # use the filtered measurement of this enzyme as features
                measurement_data[('feature',measurement)] = filtered_measurement # YOUR CODE HERE
            if measurement in target_list:
                # use the filtered measurment of this metabolite as a feature
                measurement_data[('feature',measurement)] = filtered_measurement # YOUR CODE HERE
                # additionally compute gradients of the filtered measurement and use it as target
                measurement_data[('target',measurement)] = np.gradient([point/delT for point in filtered_measurement]) # YOUR CODE HERE
   
        # Create a table
        strain_data = pd.DataFrame(measurement_data,
                                   index=pd.MultiIndex.from_product([[strain],np.linspace(minT,maxT,n_dim)],
                                   names=['Strain', 'Time']))
        ml_data = pd.concat([ml_data,strain_data])
        
    return ml_data
