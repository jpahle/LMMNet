# One-step Learning
# Wasserstein Distance, KL Divergence, Dynamic Time Warping, MSE
# 2D Harmonic Oscillator

import harmonic
from utils import *
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import harmonic
from lmmNet import *
import train_onestep
import predict_onestep
import predict_lmmNet
import train_lmmNet
from scipy.integrate import odeint
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def report_harmonic_onestep(metric_function):
    
    error_list = []
    
    time_points, test_data = harmonic.simulate_custom(xinit=1, yinit=0)

    feature_list = [] # here we do not have external time-series or control variables
    target_list = ['x_component', 'y_component']
    df_test = create_data(test_data, time_points, cols=target_list, num=2)

    for _ in range(10):
        # generate data with random initial conditions
        xi = np.random.uniform(0, 4, 2)
        yi = np.random.uniform(0, 4, 2)
        time_points, data1 = harmonic.simulate_custom(xinit=xi[0], yinit=yi[0])    
        time_points, data2 = harmonic.simulate_custom(xinit=xi[1], yinit=yi[1])

        # now generate and augment the training dataset
        df1 = create_data(data1, time_points, cols=target_list, num=0)
        df2 = create_data(data2, time_points, cols=target_list, num=1)
        df = pd.concat([df1, df2])
        df_train = train_onestep.generate_dataset(df, [0,1],feature_list, target_list, n_dim=30000)

        rf_model = RandomForestRegressor(n_estimators=20)
        figure_path = './plots/'
        rf_dict, score_dict = train_onestep.train_classic(df_train, rf_model, plot=False,model_type='random_forest', figure_path=figure_path)
        time_points, predictions = predict_onestep.predict_integrate(df_test, df, rf_dict, target_list, feature_list, title='test', plot=False,model_type='random_forest', subplots=(2,1), bio=False)
        predictions = predictions.to_numpy()

        if metric_function == "wasserstein":
            e1 = wasserstein_distance(predictions[:,0], test_data[0,:,0])
            e2 = wasserstein_distance(predictions[:,1], test_data[0,:,1])
            
        elif metric_function == "dtw":
            e1, _ = fastdtw(predictions[:,0], test_data[0,:,0], dist=euclidean)
            e2, _ = fastdtw(predictions[:,1], test_data[0,:,1], dist=euclidean)
            
        elif metric_function == "mse":
            e1 = predict_lmmNet.compute_MSE(predictions, test_data[0], 0)
            e2 = predict_lmmNet.compute_MSE(predictions, test_data[0], 1)
        error_list.append((e1, e2))
        
        # plot
        plt.plot(time_points, test_data[0,:,0], 'r.', label='x_1')
        plt.plot(time_points, test_data[0,:,1], 'y.', label='x_2')
        plt.plot(time_points, predictions[:,0], 'b--', label='predicted x_1')
        plt.plot(time_points, predictions[:,1], 'b--', label='predicted x_2')
        plt.title(str(xi) + " " + str(yi))
        plt.legend()
        plt.show()
        
    return error_list