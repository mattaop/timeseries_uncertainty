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
from pmdarima.arima import auto_arima, arima

from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_autoencoder
from src.utility.compute_coverage import compute_coverage
from src.modeling.pre_traning_autoencoder import pre_training
from src.processing.split_data import train_test_split, split_sequence


def plot_predictions(df, mean, mse, quantile_80, quantile_95, cfg):
    x_data = np.linspace(1, len(df), len(df))
    pred_sarima, pred_es = baseline_models(df, cfg)
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
                                    trend='add', seasonal='add')
    model_es = model_es.fit(optimized=True)
    pred_es = model_es.predict(start=df.index[-len(test)], end=df.index[-1])
    auto_model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3,
                            m=12, start_P=1, start_Q=1, seasonal=True, d=1, D=1, suppress_warnings=True,
                            stepwise=True)
    print('Auto arima', auto_model.aic())
    print(auto_model.summary())
    forecast_arima = auto_model.predict(n_periods=len(test),
                                        return_conf_int=True, alpha=0.05)
    pred_arima = forecast_arima[0]
    conf_int_95_arima = forecast_arima[1]
    forecast_arima = auto_model.predict(n_periods=len(test),
                                        return_conf_int=True, alpha=0.2)
    conf_inf_80_arima = forecast_arima[1]

    coverage_95pi = compute_coverage(upper_limits=conf_int_95_arima[:, 1],
                                     lower_limits=conf_int_95_arima[:, 0],
                                     actual_values=test)
    coverage_80pi = compute_coverage(upper_limits=conf_inf_80_arima[:, 1],
                                     lower_limits=conf_inf_80_arima[:, 0],
                                     actual_values=test)
    print('ARIMA 95%-prediction interval coverage: ', coverage_95pi)
    print('ARIMA 80%-prediction interval coverage: ', coverage_80pi)

    # model_arima = SARIMAX(df['y'].iloc[:-test_size], order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))
    # model_arima = model_arima.fit(full_output=False, disp=False)
    # print('Not auto arima', model_arima.aic)
    # pred_arima = model_arima.predict(start=df.index[int(len(df)*(1-cfg['test_size'])+1)], end=df.index[-1])
    # forecast_arima = model_arima.forecast(steps=int(len(df)*cfg['test_size']))

    pred_es = np.asarray(pred_es)
    pred_arima = np.asarray(pred_arima)

    x_data = np.linspace(1, len(df), len(df))
    x_predictions = np.linspace(len(train)+1, len(df), len(test))
    plt.figure()
    plt.title("SARIMA Time Series Forecasting")
    plt.plot(x_data, df, label='Data')
    plt.plot(x_predictions, pred_arima, label='SARIMA')
    plt.plot(x_predictions, pred_es, label='Exponential Smoothing')

    plt.fill_between(x_predictions, conf_inf_80_arima[:, 0], conf_inf_80_arima[:, 1],
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(x_predictions, conf_int_95_arima[:, 0], conf_int_95_arima[:, 1],
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()
    return pred_arima, pred_es


# forecast with a pre-fit model
def forecast(model, history, cfg):
    # prepare data
    x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))

    # forecast
    monte_carlo_samples = model.predict(x_input)
    return monte_carlo_samples[0]


def monte_carlo_forecast(train, test, model, cfg, encoder=None):
    prediction_sequence = np.zeros([cfg['number_of_mc_forward_passes'], len(test), cfg['forecasting_horizon']])
    if encoder:
        enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    # Number of MC samples
    print("=== Forwarding", cfg['number_of_mc_forward_passes'], "passes ===")
    for j in tqdm.tqdm(range(cfg['number_of_mc_forward_passes'])):
        history = [x for x in train]
        # Prediction horizon / test length
        for i in range(len(test)):
            # fit model and make forecast for history
            x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
            mc_sample = func([x_input, cfg['dropout_rate_test']])[0]
            # store forecast in list of predictions
            if cfg['multi_step_prediction']:
                history.append(mc_sample[0, 0])
            else:
                history.append(test[i])
            prediction_sequence[j, i] = mc_sample
    return prediction_sequence


def sliding_monte_carlo_forecast(train, test, model, cfg, inherent_noise):
    window_length = int(cfg['sequence_length']/4)
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
        print(mse_sliding)
        coverage_95_pi.append(compute_coverage(upper_limits=mean + 1.96*total_uncertainty,
                                               lower_limits=mean - 1.96*total_uncertainty,
                                               actual_values=test[i:len(test)-window_length+i]))
        width_95_pi.append(2*1.96*np.mean(total_uncertainty, axis=0)[0])

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
    print(y_hat.shape)
    print(val.shape)
    """
    plt.figure()
    plt.plot(np.linspace(1, len(val), len(val)), val, label='Val')
    plt.plot(np.linspace(1, len(val), len(val)), y_hat, label='Pred')
    plt.legend()
    plt.show()
    """

    for i in range(cfg['forecasting_horizon']):
        inherent_noise[i] = mean_squared_error(val[i:], y_hat[:-i or None, 0])
    print('Validation mse: ', inherent_noise)
    # Predict sequence over testing set using Monte Carlo dropout with n forward passes
    # train_x, train_y = split_sequence(train_and_val, cfg)
    # model = train_model(train_x, train_y, cfg)

    # coverage_95_pi_sliding_window, width_95_pi_sliding_window, mse_sliding_window = sliding_monte_carlo_forecast(train_and_val, test, model, cfg, inherent_noise)
    prediction_sequence = monte_carlo_forecast(train_and_val, test, model, cfg)

    # Compute mean and uncertainty for the Monte Carlo estimates
    mc_mean = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    mc_uncertainty = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    for i in range(cfg['forecasting_horizon']):
        mc_mean[:, i] = np.mean(prediction_sequence[:, :, i], axis=0)
        mc_uncertainty[:, i] = np.var(prediction_sequence[:, :, i], axis=0)
    # Add inherent noise and uncertainty obtained from Monte Carlo samples
    print(np.mean(inherent_noise))
    print(np.mean(mc_uncertainty))
    total_uncertainty = np.sqrt(inherent_noise + mc_uncertainty)

    # estimate prediction error
    mse = mean_squared_error(test, mc_mean)
    print(' > %.5f' % mse)

    # Compute quantiles of the Monte Carlo estimates
    for i in range(cfg['forecasting_horizon']):
        coverage_80pi = compute_coverage(upper_limits=mc_mean[:, i]+1.28*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.28*total_uncertainty[:, i],
                                         actual_values=test)
        coverage_95pi = compute_coverage(upper_limits=mc_mean[:, i]+1.96*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.96*total_uncertainty[:, i],
                                         actual_values=test)
        # print('80%-prediction interval coverage: ', i, coverage_80pi)
        # print('95%-prediction interval coverage: ', i, coverage_95pi)

    return prediction_sequence, mc_mean, total_uncertainty, mse, coverage_80pi, coverage_95pi, inherent_noise


def run_multiple_neural_networks(df, cfg):
    n_runs = 1
    mse_list, val_mse_list, coverage_80pi_list, coverage_95pi_list = [], [], [], []
    for i in range(n_runs):
        predictions, mc_mean, total_uncertainty, mse, coverage_80pi, coverage_95pi, val_mse = pipeline(df['y'].values.reshape(-1, 1), cfg)
        mse_list.append(mse)
        val_mse_list.append(val_mse)
        coverage_80pi_list.append(coverage_80pi)
        coverage_95pi_list.append(coverage_95pi)
    # print('Validation MSE', np.mean(total_uncertainty**2))
    print('MSE', mse_list)
    print('Val MSE', val_mse_list)
    print('Best model', mse_list[int(np.argmin(val_mse_list))])
    print('Best coverage', coverage_95pi_list[int(np.argmin(val_mse_list))])
    print('95%-prediction interval coverage: ', coverage_95pi_list)
    print('Average 80%-prediction interval coverage: ', np.mean(coverage_80pi_list))
    print('Average 95%-prediction interval coverage: ', np.mean(coverage_95pi_list))

    return predictions, mc_mean, total_uncertainty, np.mean(mse_list)


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

    print(df)
    scaler = MinMaxScaler()
    # df['y'] = np.log(df['y'])
    df['y'] = scaler.fit_transform(df['y'].values.reshape(-1, 1))
    if cfg['differencing']:
        df = df.diff(periods=1).dropna()
        df = df.diff(periods=12).dropna()

    predictions, mc_mean, total_uncertainty, average_mse = run_multiple_neural_networks(df, cfg)

    plot_predictions(df, mc_mean, average_mse,
                     np.array([mc_mean - 1.28*total_uncertainty, mc_mean + 1.28*total_uncertainty]),
                     np.array([mc_mean - 1.96*total_uncertainty, mc_mean + 1.96*total_uncertainty]), cfg)
    print(cfg)


if __name__ == '__main__':
    main()
