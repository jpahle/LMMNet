# One-step Learning
# Wasserstein Distance, KL Divergence, Dynamic Time Warping, MSE
# 2D Harmonic Oscillator

import harmonic
from utils import *
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import harmonic
import linear
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

def compute_average(elist):
    
    """
    Helper function to compute the average of the error report
    """
    
    avg_list = []
    for etuple in elist:
        avg_list.append(np.mean(etuple))
    mean_avg = np.mean(avg_list)
    return avg_list, mean_avg

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
            e1 /= np.linalg.norm(test_data[0,:,0], 2)**2
            e2 /= np.linalg.norm(test_data[0,:,1], 2)**2
            
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




def report_harmonic_lmmnet(metric_function):
    
    error_list = []
    
    time_points, test_data = harmonic.simulate_custom(xinit=1, yinit=0)


    for _ in range(10):
        # generate data with random initial conditions
        xi = np.random.uniform(0, 4, 1)[0]
        yi = np.random.uniform(0, 4, 1)[0]
        time_points, cubic_data = harmonic.simulate_custom(xinit=xi, yinit=yi)

        model = train_lmmNet.train_easy(time_points, cubic_data)
        x0 = test_data[0,0,:] # initial conditions
        predicted_traj = odeint(lambda x, t: predict_lmmNet.predict_fn(x, t, model), x0, time_points)

        predictions = predicted_traj
        if metric_function == "wasserstein":
            e1 = wasserstein_distance(predictions[:,0], test_data[0,:,0])
            e2 = wasserstein_distance(predictions[:,1], test_data[0,:,1])
            
        elif metric_function == "dtw":
            e1, _ = fastdtw(predictions[:,0], test_data[0,:,0], dist=euclidean)
            e2, _ = fastdtw(predictions[:,1], test_data[0,:,1], dist=euclidean)
            e1 /= np.linalg.norm(test_data[0,:,0], 2)**2
            e2 /= np.linalg.norm(test_data[0,:,1], 2)**2
            
        elif metric_function == "mse":
            e1 = predict_lmmNet.compute_MSE(predictions, test_data[0], 0)
            e2 = predict_lmmNet.compute_MSE(predictions, test_data[0], 1)
        error_list.append((e1, e2))
        
        # plot
        plt.plot(time_points, test_data[0,:,0], 'r.', label='x_1')
        plt.plot(time_points, test_data[0,:,1], 'y.', label='x_2')
        plt.plot(time_points, predictions[:,0], 'b--', label='predicted x_1')
        plt.plot(time_points, predictions[:,1], 'b--')
        plt.title(str(xi) + " " + str(yi))
        plt.legend()
        plt.show()
        
    return error_list


def report_linear_lmmnet(metric_function):
    
    error_list = []
    
    time_points, test_data = linear.simulate_default()


    for _ in range(10):
        # generate data with random initial conditions
        xi = np.random.uniform(1, 4, 1)[0]
        yi = np.random.uniform(0, 4, 1)[0]
        zi = np.random.uniform(1, 2, 1)[0]
        time_points, cubic_data = linear.simulate_custom(xinit=xi, yinit=yi, zinit=zi)

        model = train_lmmNet.train_easy(time_points, cubic_data)
        x0 = test_data[0,0,:] # initial conditions
        predicted_traj = odeint(lambda x, t: predict_lmmNet.predict_fn(x, t, model), x0, time_points)

        predictions = predicted_traj
        if metric_function == "wasserstein":
            e1 = wasserstein_distance(predictions[:,0], test_data[0,:,0])
            e2 = wasserstein_distance(predictions[:,1], test_data[0,:,1])
            e3 = wasserstein_distance(predictions[:,2], test_data[0,:,2])
            
        elif metric_function == "dtw":
            e1, _ = fastdtw(predictions[:,0], test_data[0,:,0], dist=euclidean)
            e2, _ = fastdtw(predictions[:,1], test_data[0,:,1], dist=euclidean)
            e3, _ = fastdtw(predictions[:,2], test_data[0,:,2], dist=euclidean)
            e1 /= np.linalg.norm(test_data[0,:,0], 2)**2
            e2 /= np.linalg.norm(test_data[0,:,1], 2)**2
            e3 /= np.linalg.norm(test_data[0,:,2], 2)**2
            
        elif metric_function == "mse":
            e1 = predict_lmmNet.compute_MSE(predictions, test_data[0], 0)
            e2 = predict_lmmNet.compute_MSE(predictions, test_data[0], 1)
            e3 = predict_lmmNet.compute_MSE(predictions, test_data[0], 2)
        error_list.append((e1, e2, e3))
        
        # plot
        plt.figure(figsize=(20, 10))
        plt.plot(time_points, test_data[0,:,0], 'r.', label='x_1')
        plt.plot(time_points, test_data[0,:,1], 'y.', label='x_2')
        plt.plot(time_points, test_data[0,:,2], 'g.', label='x_3')
        plt.plot(time_points, predictions[:,0], 'b--', label='predicted dynamics')
        plt.plot(time_points, predictions[:,1], 'b--')
        plt.plot(time_points, predictions[:,2], 'b--')
        plt.title(str(xi) + " " + str(yi) + " " + str(zi))
        plt.legend()
        plt.show()
        
    return error_list



def report_linear_onestep(metric_function):
    
    error_list = []
    
    time_points, test_data = linear.simulate_default()

    feature_list = [] # here we do not have external time-series or control variables
    target_list = ['x_component', 'y_component', 'z_component']
    df_test = create_data(test_data, time_points, cols=target_list, num=2)


    for _ in range(10):
        # generate data with random initial conditions
        xi = np.random.uniform(1, 4, 1)[0]
        yi = np.random.uniform(0, 4, 1)[0]
        zi = np.random.uniform(1, 2, 1)[0]
        time_points, cubic_data = linear.simulate_custom(xinit=xi, yinit=yi, zinit=zi)

        df = create_data(cubic_data, time_points, cols=target_list, num=0)
        df_train = train_onestep.generate_dataset(df, [0],feature_list, target_list, n_dim=2500)
        
        rf_model = RandomForestRegressor(n_estimators=20)
        figure_path = './plots/'
        rf_dict, score_dict = train_onestep.train_classic(df_train, rf_model, plot=False,model_type='random_forest', figure_path=figure_path)
        time_points, predictions = predict_onestep.predict_integrate(df_test, df, rf_dict, target_list, feature_list, title='test', plot=False,model_type='random_forest', subplots=(3,1), bio=False)
        predictions = predictions.to_numpy()
        
        if metric_function == "wasserstein":
            e1 = wasserstein_distance(predictions[:,0], test_data[0,:,0])
            e2 = wasserstein_distance(predictions[:,1], test_data[0,:,1])
            e3 = wasserstein_distance(predictions[:,2], test_data[0,:,2])
            
        elif metric_function == "dtw":
            e1, _ = fastdtw(predictions[:,0], test_data[0,:,0], dist=euclidean)
            e2, _ = fastdtw(predictions[:,1], test_data[0,:,1], dist=euclidean)
            e3, _ = fastdtw(predictions[:,2], test_data[0,:,2], dist=euclidean)
            e1 /= np.linalg.norm(test_data[0,:,0], 2)**2
            e2 /= np.linalg.norm(test_data[0,:,1], 2)**2
            e3 /= np.linalg.norm(test_data[0,:,2], 2)**2
            
        elif metric_function == "mse":
            e1 = predict_lmmNet.compute_MSE(predictions, test_data[0], 0)
            e2 = predict_lmmNet.compute_MSE(predictions, test_data[0], 1)
            e3 = predict_lmmNet.compute_MSE(predictions, test_data[0], 2)
        error_list.append((e1, e2, e3))
        
        # plot
        plt.figure(figsize=(20, 10))
        plt.plot(time_points, test_data[0,:,0], 'r.', label='x_1')
        plt.plot(time_points, test_data[0,:,1], 'y.', label='x_2')
        plt.plot(time_points, test_data[0,:,2], 'g.', label='x_3')
        plt.plot(time_points, predictions[:,0], 'b--', label='predicted dynamics')
        plt.plot(time_points, predictions[:,1], 'b--')
        plt.plot(time_points, predictions[:,2], 'b--')
        plt.title(str(xi) + " " + str(yi) + " " + str(zi))
        plt.legend()
        plt.show()
        
    return error_list


def report_harmonic_sindy():
    error_list = []
    
    for _ in range(10):
        error_dict = {}
        xi = np.random.uniform(-4, 4, 1)[0]
        yi = np.random.uniform(-4, 4, 1)[0]
        time_points, test_data = harmonic.simulate_custom(xinit=xi, yinit=yi)

        t0, T, h = 0, 25, .01
        _, pred = train_lmmNet.create_training_data(t0, T, h, sindy_harmonic, test_data[0,0,:])
        predictions = pred[0]
        e1 = wasserstein_distance(predictions[:,0], test_data[0,:,0])
        e2 = wasserstein_distance(predictions[:,1], test_data[0,:,1])
        error_dict['wasserstein'] = (e1, e2)

        e1, _ = fastdtw(predictions[:,0], test_data[0,:,0], dist=euclidean)
        e2, _ = fastdtw(predictions[:,1], test_data[0,:,1], dist=euclidean)
        e1 /= np.linalg.norm(test_data[0,:,0], 2)**2
        e2 /= np.linalg.norm(test_data[0,:,1], 2)**2

        error_dict['dtw'] = (e1, e2)

        e1 = predict_lmmNet.compute_MSE(predictions, test_data[0], 0)
        e2 = predict_lmmNet.compute_MSE(predictions, test_data[0], 1)
        error_dict['mse'] = (e1, e2)

        # plot
        plt.plot(time_points, test_data[0,:,0], 'r.', label='x_1')
        plt.plot(time_points, test_data[0,:,1], 'y.', label='x_2')
        plt.plot(time_points, predictions[:,0], 'b--', label='predicted x_1')
        plt.plot(time_points, predictions[:,1], 'b--')
        plt.legend()
        plt.show()
        
        error_list.append(error_dict)
        
    return error_list

def report_linear_sindy():
    
    error_list = []
    
    for _ in range(10):
    
        error_dict = {}
        xi = np.random.uniform(-4, 4, 1)[0]
        yi = np.random.uniform(-4, 4, 1)[0]
        zi = np.random.uniform(0, 2, 1)[0]
        time_points, test_data = linear.simulate_custom(xinit=xi, yinit=yi, zinit=zi)

        t0, T, h = 0, 50, .01
        _, pred = train_lmmNet.create_training_data(t0, T, h, sindy_linear, test_data[0,0,:])
        predictions = pred[0]
        e1 = wasserstein_distance(predictions[:,0], test_data[0,:,0])
        e2 = wasserstein_distance(predictions[:,1], test_data[0,:,1])
        e3 = wasserstein_distance(predictions[:,2], test_data[0,:,2])
        error_dict['wasserstein'] = (e1, e2, e3)

        e1, _ = fastdtw(predictions[:,0], test_data[0,:,0], dist=euclidean)
        e2, _ = fastdtw(predictions[:,1], test_data[0,:,1], dist=euclidean)
        e3, _ = fastdtw(predictions[:,2], test_data[0,:,2], dist=euclidean)
        e1 /= np.linalg.norm(test_data[0,:,0], 2)**2
        e2 /= np.linalg.norm(test_data[0,:,1], 2)**2
        e3 /= np.linalg.norm(test_data[0,:,2], 2)**2

        error_dict['dtw'] = (e1, e2, e3)

        e1 = predict_lmmNet.compute_MSE(predictions, test_data[0], 0)
        e2 = predict_lmmNet.compute_MSE(predictions, test_data[0], 1)
        e3 = predict_lmmNet.compute_MSE(predictions, test_data[0], 2)
        error_dict['mse'] = (e1, e2, e3)

        # plot
        plt.plot(time_points, test_data[0,:,0], 'r.', label='x_1')
        plt.plot(time_points, test_data[0,:,1], 'y.', label='x_2')
        plt.plot(time_points, test_data[0,:,2], 'g.', label='x_3')
        plt.plot(time_points, predictions[:,0], 'b--', label='predicted x_1')
        plt.plot(time_points, predictions[:,1], 'b--')
        plt.plot(time_points, predictions[:,2], 'b--')
        plt.legend()
        plt.show()

        error_list.append(error_dict)
        
    return error_list

# SINDy results

def sindy_harmonic(x,t):
    """
    The dynamics of cubic oscillator identified by SINDy
    
    The parameters are taken directly from the original paper (Brunton 2015).
    """
    A = np.array([[-0.0996, 1.997], [-1.9994,-0.0979]]) # 2 x 2

    return np.ravel(np.matmul(A,x.reshape(-1, 1)**3))

def sindy_linear(x,t):
    """
    The dynamics of linear oscillator identified by SINDy
    The parameters are taken directly from the original paper (Brunton 2015).
    """
    A = np.array([[-0.0996, -2.0005, 0], [1.9997,-0.0994, 0], [0, 0, -0.3003]]) # 3 x 3

    return np.ravel(np.matmul(A,x.reshape(-1, 1)))