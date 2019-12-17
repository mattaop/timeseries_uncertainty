import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import random as rn
import tensorflow as tf

seed = 1
rn.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from keras import backend as k
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count={'CPU': 1})
sess = tf.Session(graph=tf.get_default_graph(), config=config)
k.set_session(sess)

import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import backend as K
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima

from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_autoencoder
from src.utility.compute_coverage import compute_coverage
from src.modeling.pre_traning_autoencoder import pre_training
from src.processing.split_data import train_test_split, split_sequence


def plot_predictions(df, mean, mse, quantile_80, quantile_95, cfg):
    x_data = np.linspace(1, len(df), len(df))
    pred_sarima, pred_es, conf_int_arima = baseline_models(df, cfg)
    print("_______________________________________")
    print("|Method                     | MSE     |")
    print("|Exponential smoothing      |", "%.5f" % mean_squared_error(pred_es, df[-len(mean[:, 0]):]), "|")
    print("|ARIMA                      |", "%.5f" % mean_squared_error(pred_sarima, df[-len(mean[:, 0]):]), "|")
    print("|Neural Network             |", "%.5f" % mse, "|")
    print("_______________________________________")

    for i in range(cfg['forecasting_horizon']):
        x_predictions = np.linspace(len(df) - len(mean[:, i]) + 1+i, len(df)+i, len(mean[:, i]))
        plt.figure()
        if cfg['model'].lower() == 'cnn':
            plt.title("Convolutional Neural Network Time Series Forecasting")
        elif cfg['model'].lower() == 'lstm':
            plt.title("LSTM Neural Network Time Series Forecasting")
        elif cfg['model'].lower() == 'rnn':
            plt.title("Recurrent Neural Network Time Series Forecasting")
        else:
            plt.title("Time Series Forecasting")
        plt.plot(x_data, df, label='Data')
        plt.plot(x_predictions, pred_sarima, label='SARIMA')
        plt.plot(x_predictions, pred_es, label='Exponential Smoothing')

        plt.plot(x_predictions, mean[:, i], label='Mean')
        plt.fill_between(x_predictions, quantile_80[0, :, i], quantile_80[1, :, i],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_predictions, quantile_95[0, :, i], quantile_95[1, :, i],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()


# Compute baseline models
def baseline_models(df, cfg):
    train, test = train_test_split(df['y'], cfg['test_size'])
    model_es = ExponentialSmoothing(train, seasonal_periods=12,
                                    trend='mul', seasonal='mul')
    model_es = model_es.fit(optimized=True)
    pred_es = model_es.predict(start=df.index[-len(test)], end=df.index[-1])

    auto_model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3,
                            m=12, start_P=1, start_Q=1, seasonal=True, d=1, D=1, suppress_warnings=True,
                            stepwise=True)

    print(auto_model.summary())

    pred_arima = np.zeros([len(test)-cfg['forecast_horizon'], cfg['forecast_horizon']])
    conf_int_arima = np.zeros([len(test)-cfg['forecast_horizon'], cfg['forecast_horizon'], 2])
    for i in range(len(test)-cfg['forecast_horizon']):
        forecast_arima = auto_model.predict(n_periods=cfg['forecast_horizon'],
                                            return_conf_int=True, alpha=0.05)
        pred_arima[i] = forecast_arima[0]
        conf_int_arima[i] = forecast_arima[1]
        auto_model.update(y=[test[i]])

        """
        t = np.linspace(1, cfg['forecast_horizon'], cfg['forecast_horizon'])
        plt.figure()
        plt.title("Time Series Forecasting")
        plt.plot(t, forecast_arima[0], label='Mean')
        plt.plot(t, test[i:cfg['forecast_horizon'] + i])
        plt.fill_between(t, forecast_arima[1][:, 0], forecast_arima[1][:, 1],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()
        """

    mse, coverage, prediction_interval_width = [], [], []
    for i in range(cfg['forecast_horizon']):

        mse.append(mean_squared_error(test[i:len(test)-cfg['forecast_horizon']+i], pred_arima[:, i]))
        coverage_95pi = compute_coverage(upper_limits=conf_int_arima[:, i, 1],
                                         lower_limits=conf_int_arima[:, i, 0],
                                         actual_values=test[i:len(test)-cfg['forecast_horizon']+i])
        coverage.append(coverage_95pi)
        prediction_interval_width.append(np.mean(conf_int_arima[:, i, 1]-conf_int_arima[:, i, 0], axis=0))

    print('ARIMA')
    print('MSE sliding window', mse)
    print('Coverage 95% PI sliding window', coverage)
    print('Width 95% PI sliding window', prediction_interval_width)
    print(np.mean(mse))
    return mse, coverage, prediction_interval_width


def sliding_monte_carlo_forecast(train, test, model, cfg, inherent_noise):
    window_length = int(cfg['forecast_horizon'])
    prediction_sequence = np.zeros([len(test)-window_length, cfg['number_of_mc_forward_passes'], window_length, cfg['forecasting_horizon']])
    func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    # Number of MC samples
    forward_validation_set = [x for x in train]
    print("=== Forwarding", cfg['number_of_mc_forward_passes'], "passes ===")
    for l in tqdm.tqdm(range(len(test)-window_length)):
        for j in range(cfg['number_of_mc_forward_passes']):
            history = [x for x in forward_validation_set]
            # Prediction horizon / test length
            for i in range(window_length):
                # fit model and make forecast for history
                x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
                mc_sample = func([x_input, cfg['dropout_rate_test']])[0]
                # store forecast in list of predictions
                if cfg['multi_step_prediction']:
                    history.append(mc_sample[0, 0])
                else:
                    history.append(test[i])
                prediction_sequence[l, j, i] = mc_sample
        forward_validation_set.append(test[l])
        """
        total_uncertainty = np.sqrt(inherent_noise + np.var(prediction_sequence[l], axis=0))
        mean = prediction_sequence[l].mean(axis=0)
        t = np.linspace(1, window_length, window_length)
        mean = mean[:, 0]
        total_uncertainty = total_uncertainty[:, 0]

        plt.figure()
        plt.title("Time Series Forecasting")
        plt.plot(t, mean, label='Mean')
        plt.plot(t, test[l:window_length+l])
        plt.fill_between(t, mean - 1.28*total_uncertainty, mean + 1.28*total_uncertainty,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(t, mean - 1.96*total_uncertainty, mean + 1.96*total_uncertainty,
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()
        """

    mse_sliding, coverage_95_pi, width_95_pi = [], [], []
    for i in range(window_length):
        total_uncertainty = np.sqrt(inherent_noise + np.var(prediction_sequence[:, :, i], axis=1))
        mean = prediction_sequence[:, :, i].mean(axis=1)
        mse_sliding.append(mean_squared_error(test[i:len(test)-window_length+i], mean))
        coverage_95_pi.append(compute_coverage(upper_limits=mean + 1.96*total_uncertainty,
                                               lower_limits=mean - 1.96*total_uncertainty,
                                               actual_values=test[i:len(test)-window_length+i]))
        width_95_pi.append(2*1.96*np.mean(total_uncertainty, axis=0)[0])
    """
    t = np.linspace(1, window_length, window_length)
    plt.figure()
    plt.plot(t, coverage_95_pi)
    plt.title('95% Prediction Interval Coverage')
    plt.xlabel('Forecast length (months)')
    plt.ylabel('Average coverage')
    plt.show()

    plt.figure()
    plt.plot(t, width_95_pi)
    plt.title('95% Prediction Interval Width')
    plt.xlabel('Forecast length (months)')
    plt.ylabel('Width of prediction interval')
    plt.show()
    """

    print(coverage_95_pi)
    print(width_95_pi)
    return coverage_95_pi, width_95_pi, mse_sliding


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def train_autoencoder(data, cfg):
    # Train autoencoder as pre training
    encoder_train = np.concatenate([data.train_conv, data.train_org], axis=0)
    encoder, decoder, cfg = build_autoencoder(encoder_train, cfg, weights='weights//pretrained_encoder.hdf5')
    return encoder, cfg


# walk-forward validation for univariate data
def pipeline(df, cfg):
    print(df)
    train_and_val, test = train_test_split(df, cfg['test_size'])
    print('Length train', len(train_and_val))
    print('Length test', len(test))

    train, val = train_test_split(train_and_val, cfg['validation_size'])
    train_x, train_y = split_sequence(train, cfg)
    val_x, val_y = split_sequence(np.concatenate([train[-cfg['sequence_length']:], val]), cfg)

    # If using an encoder, extract features from training data,
    model = train_model(train_x, train_y, cfg, val_x, val_y)

    # Compute inherent noise on validation set
    history = [x for x in train]
    y_hat = np.zeros([len(val), train_y.shape[1]])
    for i in range(len(val)):
        x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
        y_hat[i] = model.predict(x_input)[0]
        history.append(val[i])

    inherent_noise = np.zeros(cfg['forecasting_horizon'])
    for i in range(cfg['forecasting_horizon']):
        inherent_noise[i] = mean_squared_error(val[i:], y_hat[:-i or None, 0])
    print('Validation mse: ', inherent_noise)

    # Predict sequence over testing set using Monte Carlo dropout with n forward passes
    coverage_95_pi_sliding_window, width_95_pi_sliding_window, mse_sliding_window = sliding_monte_carlo_forecast(train_and_val, test, model, cfg, inherent_noise)

    return coverage_95_pi_sliding_window, width_95_pi_sliding_window, mse_sliding_window


def run_multiple_neural_networks(df, cfg):
    n_runs = 5
    coverage_95_pi = np.zeros([n_runs, cfg['forecast_horizon']])
    width_95_pi = np.zeros([n_runs, cfg['forecast_horizon']])
    mse = np.zeros([n_runs, cfg['forecast_horizon']])
    if cfg['encoder']:
        pre_training(df, cfg)
    for i in range(n_runs):
        coverage_95_pi_sliding_window, width_95_pi_sliding_window, mse_sliding_window = pipeline(df['y'].values.reshape(-1, 1), cfg)
        coverage_95_pi[i] = coverage_95_pi_sliding_window
        width_95_pi[i] = width_95_pi_sliding_window
        mse[i] = mse_sliding_window
    mean_mse = np.mean(mse, axis=0)
    mean_coverage = np.mean(coverage_95_pi, axis=0)
    mean_width = np.mean(width_95_pi, axis=0)
    print('-----------------------------------------------------------')
    print('MSE sliding window', mean_mse)
    print('Coverage 95% PI sliding window', mean_coverage)
    print('Width 95% PI sliding window', mean_width)
    print('Average MSE 95% PI', np.mean(mean_mse))
    print('Average coverage 95% PI', np.mean(mean_coverage))
    print('Average width 95% PI', np.mean(mean_width))

    t = np.linspace(1, cfg['forecast_horizon'], cfg['forecast_horizon'])

    plt.figure()
    plt.plot(t, mean_mse)
    plt.title('Forecasting MSE')
    plt.xlabel('Forecast length (months)')
    plt.ylabel('MSE')
    plt.show()

    plt.figure()
    plt.plot(t, mean_coverage)
    plt.title('95% Prediction Interval Coverage')
    plt.xlabel('Forecast length (months)')
    plt.ylabel('Average coverage')
    plt.show()

    plt.figure()
    plt.plot(t, mean_width)
    plt.title('95% Prediction Interval Width')
    plt.xlabel('Forecast length (months)')
    plt.ylabel('Width of prediction interval')
    plt.show()


def main():
    df, cfg = load_data()

    """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(df)
    plt.xlabel('Year')
    plt.ylabel('Number of air passengers (in 1000s)')
    plt.title('Air Passengers Data (1949-1960)')
    ax.set_xticks(df.index[::12])
    ax.set_xticklabels(df.index[::12])
    plt.show()
    """

    scaler = MinMaxScaler()
    # df['y'] = np.log(df['y'])
    df['y'] = scaler.fit_transform(df['y'].values.reshape(-1, 1))

    if cfg['differencing']:
        df = df.diff(periods=1).dropna()
        df = df.diff(periods=12).dropna()

    run_multiple_neural_networks(df, cfg)
    baseline_models(df, cfg)
    print(cfg)


if __name__ == '__main__':
    main()
