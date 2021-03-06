import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import random as rn
import tensorflow as tf
import scipy.stats as st

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
from statsmodels.tsa.stattools import adfuller

from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_autoencoder
from src.utility.compute_coverage import compute_coverage
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


# model
def pre_training(train_and_val, cfg):
    # Train autoencoder as pre training
    # train_and_val, test = train_test_split(df, cfg['test_size'])
    # scaler = MinMaxScaler()
    # train_and_val = scaler.fit_transform(train_and_val.reshape(-1, 1))
    # train_and_val[train_and_val.columns.values] = scaler.fit_transform(train_and_val[train_and_val.columns.values].values)
    train, val = train_test_split(train_and_val, cfg['validation_size'])

    train_x = []
    train_y = []
    validation_x = []
    validation_y = []
    for name, values in train.iteritems():
        x, y = split_sequence(values.values.reshape(-1, 1), cfg)
        val_x, val_y = split_sequence(np.concatenate([values[-cfg['sequence_length']:], val[name]]).reshape(-1, 1), cfg)
        train_x.append(x)
        train_y.append(y)
        validation_x.append(val_x)
        validation_y.append(val_y)
    train_x = np.asarray(train_x)
    train_x = train_x.reshape([train_x.shape[0]*train_x.shape[1], train_x.shape[2], train_x.shape[3]])
    train_y = np.asarray(train_y)
    train_y = train_y.reshape([train_y.shape[0] * train_y.shape[1], train_y.shape[2]])

    validation_x = np.asarray(validation_x)
    validation_x = validation_x.reshape([validation_x.shape[0] * validation_x.shape[1], validation_x.shape[2], validation_x.shape[3]])
    validation_y = np.asarray(validation_y)
    validation_y = validation_y.reshape([validation_y.shape[0] * validation_y.shape[1], validation_y.shape[2]])

    # encoder, decoder, cfg = build_autoencoder(train_x, cfg, weights='weights//pretrained_encoder.hdf5')

    # If using an encoder, extract features from training data,
    model = train_model(train_x, train_y, cfg, validation_x, validation_y)

    # Test autoencoder on holdout data
    #encoder_test = data.x_test
    #predictions = decoder.predict(encoder_test)

    #mse = 0
    #for i in range(len(predictions)):
    #    mse += mean_squared_error(encoder_test[i], predictions[i])
    #print('Test mean mse:', mse/len(predictions))

    return model


def exponential_smoothing(df, cfg):
    train, test = train_test_split(df['y'], cfg['test_size'])

    # scaler = MinMaxScaler()
    # train['y'] = scaler.fit_transform(train.values.reshape(-1, 1))
    # test['y'] = scaler.transform(test.values.reshape(-1, 1))
    trends = [None, 'add', 'add_damped']
    seasons = [None, 'add', 'mul']
    best_model_parameters = [None, None, False]  # trend, season, damped
    best_aicc = np.inf
    for trend in trends:
        for season in seasons:
            if trend == 'add_damped':
                trend = 'add'
                damped = True
            else:
                damped = False
            model_es = ExponentialSmoothing(train, seasonal_periods=52,
                                            trend=trend, seasonal=season,
                                            damped=damped)
            model_es = model_es.fit(optimized=True)
            if model_es.aicc < best_aicc:
                best_model_parameters = [trend, season, damped]
                best_aicc = model_es.aicc
    model_es = ExponentialSmoothing(train, seasonal_periods=52,
                                    trend=best_model_parameters[0], seasonal=best_model_parameters[1],
                                    damped=best_model_parameters[2])
    model_es = model_es.fit(optimized=True)
    print('ETS: T=', best_model_parameters[0], ', S=', best_model_parameters[1], ', damped=', best_model_parameters[2])
    print('AICc', model_es.aicc)
    residual_variance = model_es.sse / len(train - 2)
    var = []
    alpha = model_es.params['smoothing_level']
    beta = model_es.params['smoothing_slope']
    gamma = model_es.params['smoothing_seasonal']
    for j in range(cfg['forecast_horizon']):
        s = 12
        h = j+1
        k = int((h-1)/s)
        theta_h = cfg['forecast_horizon']
        #var.append(residual_variance*(1+j*alpha**2*(1+(j+1)*beta+(j+1)/6*(2*(j+1)-1)*beta**2)
        #                              + 12*(gamma**2*(1-alpha)**2)+alpha*gamma*(1-alpha)*(2+cfg['forecast_horizon']*beta*13)))
        #var.append(residual_variance*(1+(h-1)*alpha**2*(1+h*beta + h/6*(2*h-1)*beta**2)
        #                              + k*(gamma**2*(1-alpha)**2 + alpha*gamma*(1-alpha)*(2+k*beta*(s+1)))))
        if best_model_parameters[1] == 'add':
            if best_model_parameters[0] == 'add':
                var.append(residual_variance * (1 + (h - 1) * (alpha**2 + alpha*h*beta + h / 6 * (2 * h - 1) * beta ** 2)
                                                + k*gamma*(2*alpha + gamma + beta*s*(k + 1))))
            else:
                var.append(
                    residual_variance * (1 + (h - 1) * alpha ** 2 + k * gamma * (2 * alpha + gamma)))
        elif best_model_parameters[1] == 'mul':
            var.append(residual_variance*h)
        else:
            if best_model_parameters[0] == 'add':
                var.append(
                    residual_variance*(1 + (h - 1)*(alpha ** 2 + alpha * h * beta + h / 6 * (2 * h - 1) * beta ** 2)))
            else:
                var.append(residual_variance * (1 + (h - 1) * alpha ** 2))
    pred_es = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon']])
    conf_int_es_80 = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon'], 2])
    conf_int_es_95 = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon'], 2])

    for i in range(len(test) - cfg['forecast_horizon']):
        pred_es[i] = model_es.forecast(steps=cfg['forecast_horizon'] + i)[-cfg['forecast_horizon']:]
        conf_int_es_80[i, :, 0] = pred_es[i] - 1.28 * np.sqrt(var)
        conf_int_es_80[i, :, 1] = pred_es[i] + 1.28 * np.sqrt(var)
        conf_int_es_95[i, :, 0] = pred_es[i] - 1.96 * np.sqrt(var)
        conf_int_es_95[i, :, 1] = pred_es[i] + 1.96 * np.sqrt(var)

    # Store results
    mse_es, coverage_es_95, coverage_es_80, width_es_95, width_es_80 = [], [], [], [], []

    for i in range(cfg['forecast_horizon']):
        # Exponential Smoothing MSE
        mse_es.append(mean_squared_error(test[i:len(test) - cfg['forecast_horizon'] + i], pred_es[:, i]))

        # Exponential Smoothing 80% PI
        coverage_es_80.append(compute_coverage(upper_limits=conf_int_es_80[:, i, 1],
                                               lower_limits=conf_int_es_80[:, i, 0],
                                               actual_values=test.values[i:len(test) - cfg['forecast_horizon'] + i]))
        width_es_80.append(np.mean(conf_int_es_80[:, i, 1] - conf_int_es_80[:, i, 0], axis=0))

        # Exponential Smoothing 95% PI
        coverage_es_95.append(compute_coverage(upper_limits=conf_int_es_95[:, i, 1],
                                               lower_limits=conf_int_es_95[:, i, 0],
                                               actual_values=test.values[i:len(test) - cfg['forecast_horizon'] + i]))
        width_es_95.append(np.mean(conf_int_es_95[:, i, 1] - conf_int_es_95[:, i, 0], axis=0))

    print('================ ES ====================')
    print('MSE sliding window', mse_es)
    print('Mean MSE', np.mean(mse_es))
    print('Coverage of 80% PI sliding window', coverage_es_80)
    print('Width of 80% PI sliding window', width_es_80)
    print('Coverage of 95% PI sliding window', coverage_es_95)
    print('Width of 95% PI sliding window', width_es_95)
    return mse_es, coverage_es_80, coverage_es_95, width_es_80, width_es_95


def arima(df, cfg):
    # df = df.diff(periods=1).dropna()
    # result = adfuller(df['y'].values)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])

    train, test = train_test_split(df['y'], cfg['test_size'])
    #auto_model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3,
    #                        m=52, start_P=1, start_Q=1, seasonal=True, d=1, D=1, suppress_warnings=True,
    #                        stepwise=True)
    auto_model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, max_d=1, max_P=1, max_Q=1, max_D=1,
                            m=52, start_P=0, start_Q=0, seasonal=True, d=None, D=None, suppress_warnings=True,
                            stepwise=True, information_criterion='aicc')

    print(auto_model.summary())

    pred_arima = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon']])
    conf_int_arima_80 = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon'], 2])
    conf_int_arima_95 = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon'], 2])

    for i in range(len(test)-cfg['forecast_horizon']):
        forecast_arima_95 = auto_model.predict(n_periods=cfg['forecast_horizon'],
                                               return_conf_int=True, alpha=1-0.95)
        forecast_arima_80 = auto_model.predict(n_periods=cfg['forecast_horizon'],
                                               return_conf_int=True, alpha=1-0.8)
        pred_arima[i] = forecast_arima_95[0]
        conf_int_arima_80[i] = forecast_arima_80[1]
        conf_int_arima_95[i] = forecast_arima_95[1]
        auto_model.update(y=[test.values[i]])

    # Store results
    mse_arima, coverage_arima_95, coverage_arima_80, width_arima_95, width_arima_80 = [], [], [], [], []

    for i in range(cfg['forecast_horizon']):
        # ARIMA mean squared error (MSE):
        mse_arima.append(mean_squared_error(test[i:len(test)-cfg['forecast_horizon']+i], pred_arima[:, i]))

        # ARIMA 80% PI
        coverage_arima_80.append(compute_coverage(upper_limits=conf_int_arima_80[:, i, 1],
                                                  lower_limits=conf_int_arima_80[:, i, 0],
                                                  actual_values=test.values[i:len(test) - cfg['forecast_horizon'] + i]))
        width_arima_80.append(np.mean(conf_int_arima_80[:, i, 1] - conf_int_arima_80[:, i, 0], axis=0))

        # ARIMA 95% PI
        coverage_arima_95.append(compute_coverage(upper_limits=conf_int_arima_95[:, i, 1],
                                                  lower_limits=conf_int_arima_95[:, i, 0],
                                                  actual_values=test.values[i:len(test)-cfg['forecast_horizon']+i]))
        width_arima_95.append(np.mean(conf_int_arima_95[:, i, 1] - conf_int_arima_95[:, i, 0], axis=0))

    print('================ ARIMA =================')
    print('Mean MSE', np.mean(mse_arima))
    print('MSE sliding window', mse_arima)
    print('Coverage of 80% PI sliding window', coverage_arima_80)
    print('Width of 80% PI sliding window', width_arima_80)
    print('Coverage of 95% PI sliding window', coverage_arima_95)
    print('Width of 95% PI sliding window', width_arima_95)
    return mse_arima, coverage_arima_80, coverage_arima_95, width_arima_80, width_arima_95


# Compute baseline models
def baseline_models(df, cfg):
    result = adfuller(df['y'].values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    # df = df.diff(periods=1).dropna()
    # result = adfuller(df['y'].values)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])

    train, test = train_test_split(df['y'], cfg['test_size'])

    # scaler = MinMaxScaler()
    # train['y'] = scaler.fit_transform(train.values.reshape(-1, 1))
    # test['y'] = scaler.transform(test.values.reshape(-1, 1))
    trends = [None, 'add', 'add_damped']
    seasons = [None, 'add', 'mul']
    best_model_parameters = [None, None, False]  # trend, season, damped
    best_aicc = np.inf
    for trend in trends:
        for season in seasons:
            if trend == 'add_damped':
                trend = 'add'
                damped = True
            else:
                damped = False
            model_es = ExponentialSmoothing(train, seasonal_periods=52,
                                            trend=trend, seasonal=season,
                                            damped=damped)
            model_es = model_es.fit(optimized=True)
            if model_es.aicc < best_aicc:
                best_model_parameters = [trend, season, damped]
                best_aicc = model_es.aicc
    model_es = ExponentialSmoothing(train, seasonal_periods=52,
                                    trend=best_model_parameters[0], seasonal=best_model_parameters[1],
                                    damped=best_model_parameters[2])
    model_es = model_es.fit(optimized=True)
    print('ETS: T=', best_model_parameters[0], ', S=', best_model_parameters[1], ', damped=', best_model_parameters[2])
    print('AICc', model_es.aicc)
    residual_variance = model_es.sse/len(train-2)
    var = []
    alpha = model_es.params['smoothing_level']
    beta = model_es.params['smoothing_slope']
    gamma = model_es.params['smoothing_seasonal']
    for j in range(cfg['forecast_horizon']):
        s = 12
        h = j+1
        k = int((h-1)/s)
        #var.append(residual_variance*(1+j*alpha**2*(1+(j+1)*beta+(j+1)/6*(2*(j+1)-1)*beta**2)
        #                              + 12*(gamma**2*(1-alpha)**2)+alpha*gamma*(1-alpha)*(2+cfg['forecast_horizon']*beta*13)))
        #var.append(residual_variance*(1+(h-1)*alpha**2*(1+h*beta + h/6*(2*h-1)*beta**2)
        #                              + k*(gamma**2*(1-alpha)**2 + alpha*gamma*(1-alpha)*(2+k*beta*(s+1)))))
        var.append(residual_variance * (1 + (h - 1) * (alpha**2 + alpha*h*beta + h / 6 * (2 * h - 1) * beta ** 2)
                                        + k*gamma*(2*alpha + gamma + beta*s*(k + 1))))

    auto_model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3,
                            m=52, start_P=1, start_Q=1, seasonal=True, d=1, D=1, suppress_warnings=True,
                            stepwise=True)

    print(auto_model.summary())

    pred_es = np.zeros([len(test)-cfg['forecast_horizon'], cfg['forecast_horizon']])
    conf_int_es_80 = np.zeros([len(test)-cfg['forecast_horizon'], cfg['forecast_horizon'], 2])
    conf_int_es_95 = np.zeros([len(test)-cfg['forecast_horizon'], cfg['forecast_horizon'], 2])

    pred_arima = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon']])
    conf_int_arima_80 = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon'], 2])
    conf_int_arima_95 = np.zeros([len(test) - cfg['forecast_horizon'], cfg['forecast_horizon'], 2])

    for i in range(len(test)-cfg['forecast_horizon']):
        forecast_arima_95 = auto_model.predict(n_periods=cfg['forecast_horizon'],
                                               return_conf_int=True, alpha=1-0.95)
        forecast_arima_80 = auto_model.predict(n_periods=cfg['forecast_horizon'],
                                               return_conf_int=True, alpha=1-0.8)
        pred_es[i] = model_es.forecast(steps=cfg['forecast_horizon']+i)[-cfg['forecast_horizon']:]
        conf_int_es_80[i, :, 0] = pred_es[i] - 1.28 * np.sqrt(var)
        conf_int_es_80[i, :, 1] = pred_es[i] + 1.28 * np.sqrt(var)
        conf_int_es_95[i, :, 0] = pred_es[i] - 1.96 * np.sqrt(var)
        conf_int_es_95[i, :, 1] = pred_es[i] + 1.96 * np.sqrt(var)
        pred_arima[i] = forecast_arima_95[0]
        conf_int_arima_80[i] = forecast_arima_80[1]
        conf_int_arima_95[i] = forecast_arima_95[1]
        auto_model.update(y=[test.values[i]])

    # Store results
    mse_arima, coverage_arima_95, coverage_arima_80, width_arima_95, width_arima_80 = [], [], [], [], []
    mse_es, coverage_es_95, coverage_es_80, width_es_95, width_es_80 = [], [], [], [], []

    for i in range(cfg['forecast_horizon']):
        # ARIMA mean squared error (MSE):
        mse_arima.append(mean_squared_error(test[i:len(test)-cfg['forecast_horizon']+i], pred_arima[:, i]))

        # ARIMA 80% PI
        coverage_arima_80.append(compute_coverage(upper_limits=conf_int_arima_80[:, i, 1],
                                                  lower_limits=conf_int_arima_80[:, i, 0],
                                                  actual_values=test.values[i:len(test) - cfg['forecast_horizon'] + i]))
        width_arima_80.append(np.mean(conf_int_arima_80[:, i, 1] - conf_int_arima_80[:, i, 0], axis=0))

        # ARIMA 95% PI
        coverage_arima_95.append(compute_coverage(upper_limits=conf_int_arima_95[:, i, 1],
                                                  lower_limits=conf_int_arima_95[:, i, 0],
                                                  actual_values=test.values[i:len(test)-cfg['forecast_horizon']+i]))
        width_arima_95.append(np.mean(conf_int_arima_95[:, i, 1] - conf_int_arima_95[:, i, 0], axis=0))

        # Exponential Smoothing MSE
        mse_es.append(mean_squared_error(test[i:len(test) - cfg['forecast_horizon'] + i], pred_es[:, i]))

        # Exponential Smoothing 80% PI
        coverage_es_80.append(compute_coverage(upper_limits=conf_int_es_80[:, i, 1],
                                               lower_limits=conf_int_es_80[:, i, 0],
                                               actual_values=test.values[i:len(test) - cfg['forecast_horizon'] + i]))
        width_es_80.append(np.mean(conf_int_es_80[:, i, 1] - conf_int_es_80[:, i, 0], axis=0))

        # Exponential Smoothing 95% PI
        coverage_es_95.append(compute_coverage(upper_limits=conf_int_es_95[:, i, 1],
                                               lower_limits=conf_int_es_95[:, i, 0],
                                               actual_values=test.values[i:len(test) - cfg['forecast_horizon'] + i]))
        width_es_95.append(np.mean(conf_int_es_95[:, i, 1] - conf_int_es_95[:, i, 0], axis=0))

    print('================ ARIMA =================')
    print('Mean MSE', np.mean(mse_arima))
    print('MSE sliding window', mse_arima)
    print('Coverage of 80% PI sliding window', coverage_arima_80)
    print('Width of 80% PI sliding window', width_arima_80)
    print('Coverage of 95% PI sliding window', coverage_arima_95)
    print('Width of 95% PI sliding window', width_arima_95)
    print('================ ES ====================')
    print('MSE sliding window', np.mean(mse_es))
    print('Mean MSE', np.mean(mse_arima))
    print('Coverage of 80% PI sliding window', coverage_es_80)
    print('Width of 80% PI sliding window', width_es_80)
    print('Coverage of 95% PI sliding window', coverage_es_95)
    print('Width of 95% PI sliding window', width_es_95)
    return mse_arima, coverage_arima_80, coverage_arima_95, width_arima_80, width_arima_95, coverage_es_80, mse_es, \
           coverage_es_95, width_es_80, width_es_95


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

    mse_sliding = []
    coverage_95_pi, width_95_pi = [], []
    coverage_80_pi, width_80_pi = [], []

    for i in range(window_length):
        total_uncertainty = np.sqrt(inherent_noise + np.var(prediction_sequence[:, :, i], axis=1))
        mean = prediction_sequence[:, :, i].mean(axis=1)
        mse_sliding.append(mean_squared_error(test[i:len(test)-window_length+i], mean))
        coverage_95_pi.append(compute_coverage(upper_limits=mean + 1.96*total_uncertainty,
                                               lower_limits=mean - 1.96*total_uncertainty,
                                               actual_values=test[i:len(test)-window_length+i]))
        coverage_80_pi.append(compute_coverage(upper_limits=mean + 1.28 * total_uncertainty,
                                               lower_limits=mean - 1.28 * total_uncertainty,
                                               actual_values=test[i:len(test) - window_length + i]))
        width_95_pi.append(2*1.96*np.mean(total_uncertainty, axis=0)[0])
        width_80_pi.append(2*1.28*np.mean(total_uncertainty, axis=0)[0])
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
    return mse_sliding, coverage_95_pi, width_95_pi, coverage_80_pi, width_80_pi


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def train_autoencoder(data, cfg):
    # Train autoencoder as pre training
    encoder_train = np.concatenate([data.train_conv, data.train_org], axis=0)
    encoder, decoder, cfg = build_autoencoder(encoder_train, cfg, weights='weights//pretrained_encoder.hdf5')
    return encoder, cfg


def pipeline_baseline(df, cfg, model='arima'):
    n_columns = len(df.columns.values)
    coverage_80_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    width_80_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    coverage_95_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    width_95_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    mse = np.zeros([n_columns, cfg['forecast_horizon']])
    train_and_val, test = train_test_split(df, cfg['test_size'])
    scaler = MinMaxScaler(feature_range=(10**(-10), 1)).fit(train_and_val)
    df[df.columns.values] = scaler.transform(df[df.columns.values].values)
    print(train_and_val)
    print(df)
    i = 0
    if cfg['differencing']:
        df = df.diff(periods=1, axis=0).dropna()
    for name, values in df.iteritems():
        single_time_series = pd.DataFrame(data=values.values.reshape(-1, 1), index=values.index, columns=['y'])
        if model == 'arima':
            mse_sliding_window, coverage_80_pi_sliding_window, coverage_95_pi_sliding_window, \
                width_80_pi_sliding_window, width_95_pi_sliding_window = arima(single_time_series, cfg)
        else:
            mse_sliding_window, coverage_80_pi_sliding_window, coverage_95_pi_sliding_window, \
                width_80_pi_sliding_window, width_95_pi_sliding_window = exponential_smoothing(single_time_series, cfg)
        # baseline_models(single_time_series, cfg)
        coverage_80_pi[i] = coverage_80_pi_sliding_window
        width_80_pi[i] = width_80_pi_sliding_window
        coverage_95_pi[i] = coverage_95_pi_sliding_window
        width_95_pi[i] = width_95_pi_sliding_window
        mse[i] = mse_sliding_window
        i += 1
    mean_mse = np.mean(mse, axis=0)
    mean_coverage_95 = np.mean(coverage_95_pi, axis=0)
    mean_coverage_80 = np.mean(coverage_80_pi, axis=0)
    mean_width_95 = np.mean(width_95_pi, axis=0)
    mean_width_80 = np.mean(width_80_pi, axis=0)
    print('------------------', model, '-------------------------')
    print('MSE sliding window', list(mean_mse))
    print('Coverage 80% PI sliding window', list(mean_coverage_80))
    print('Width 80% PI sliding window', list(mean_width_80))
    print('Coverage 95% PI sliding window', list(mean_coverage_95))
    print('Width 95% PI sliding window', list(mean_width_95))
    print('Average MSE', np.mean(mean_mse))
    print('Average coverage 80% PI', np.mean(mean_coverage_80))
    print('Average width 80% PI', np.mean(mean_width_80))
    print('Average coverage 95% PI', np.mean(mean_coverage_95))
    print('Average width 95% PI', np.mean(mean_width_95))


# walk-forward validation for univariate data
def pipeline(train_and_val, test, cfg, model=None):
    # train_and_val, test = train_test_split(df, cfg['test_size'])
    # scaler = MinMaxScaler()
    # train_and_val = scaler.fit_transform(train_and_val.reshape(-1, 1))
    # test = scaler.transform(test.reshape(-1, 1))
    # print('Length train', len(train_and_val))
    # print('Length test', len(test))

    train, val = train_test_split(train_and_val, cfg['validation_size'])
    print(len(train), len(val), len(test))

    train_x, train_y = split_sequence(train, cfg)
    val_x, val_y = split_sequence(np.concatenate([train[-cfg['sequence_length']:], val]), cfg)

    # If using an encoder, extract features from training data,
    if not model:
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
    mse_sliding, coverage_95_pi, width_95_pi, coverage_80_pi, width_80_pi = sliding_monte_carlo_forecast(train_and_val,
                                                                                                         test, model,
                                                                                                         cfg,
                                                                                                         inherent_noise)
    print('Mean mse', np.mean(mse_sliding))
    return mse_sliding, coverage_95_pi, width_95_pi, coverage_80_pi, width_80_pi


def run_multiple_neural_networks(df, cfg):
    # Scale data
    train_and_val, test = train_test_split(df, cfg['test_size'])
    scaler = MinMaxScaler().fit(train_and_val)
    df[df.columns.values] = scaler.transform(df[df.columns.values].values)
    if cfg['differencing']:
        df = df.diff(periods=1, axis=0).dropna()
    train_and_val, test = train_test_split(df, cfg['test_size'])

    # train_and_val = scaler.fit_transform(train_and_val.reshape(-1, 1))
    #train_and_val[train_and_val.columns.values] = scaler.fit_transform(
    #    train_and_val[train_and_val.columns.values].values)
    #test[test.columns.values] = scaler.transform(test[test.columns.values].values)

    n_columns = len(df.columns.values)
    coverage_80_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    width_80_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    coverage_95_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    width_95_pi = np.zeros([n_columns, cfg['forecast_horizon']])
    mse = np.zeros([n_columns, cfg['forecast_horizon']])
    i = 0
    if cfg['autoencoder']:
        model = pre_training(train_and_val, cfg)
    else:
        model = None
    for columnName, columnData in train_and_val.iteritems():
        mse_sliding_window, coverage_95_pi_sliding_window, width_95_pi_sliding_window, coverage_80_pi_sliding_window, \
            width_80_pi_sliding_window = pipeline(train_and_val[columnName].values.reshape(-1, 1),
                                                  test[columnName].values.reshape(-1, 1), cfg, model)
        coverage_80_pi[i] = coverage_80_pi_sliding_window
        width_80_pi[i] = width_80_pi_sliding_window
        coverage_95_pi[i] = coverage_95_pi_sliding_window
        width_95_pi[i] = width_95_pi_sliding_window
        mse[i] = mse_sliding_window
        i += 1
    mean_mse = np.mean(mse, axis=0)
    mean_coverage_95 = np.mean(coverage_95_pi, axis=0)
    mean_coverage_80 = np.mean(coverage_80_pi, axis=0)
    mean_width_95 = np.mean(width_95_pi, axis=0)
    mean_width_80 = np.mean(width_80_pi, axis=0)
    print('-----------------------------------------------------------')
    print('MSE sliding window', list(mean_mse))
    print('Coverage 95% PI sliding window', list(mean_coverage_95))
    print('Width 95% PI sliding window', list(mean_width_95))
    print('Coverage 80% PI sliding window', list(mean_coverage_80))
    print('Width 80% PI sliding window', list(mean_width_80))

    print('Average MSE', np.mean(mean_mse))
    print('Average coverage 95% PI', np.mean(mean_coverage_95))
    print('Average width 95% PI', np.mean(mean_width_95))
    print('Average coverage 80% PI', np.mean(mean_coverage_80))
    print('Average width 80% PI', np.mean(mean_width_80))


def main():
    df, cfg = load_data(data_set='avocado')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.loc[:, ('AveragePrice', 'region', 'type')]
    df = df.pivot_table(index='Date', columns=['region', 'type'], aggfunc='mean')
    df = df.fillna(method='backfill').dropna()
    df.sort_index(inplace=True)

    #if cfg['differencing']:
    #    df = df.diff(periods=1, axis=0).dropna()
        # df = df.diff(periods=12).dropna()

    run_multiple_neural_networks(df, cfg)
    #pipeline_baseline(df, cfg, model='es')
    #pipeline_baseline(df, cfg, model='arima')

    print(cfg)


if __name__ == '__main__':
    main()
