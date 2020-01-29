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

import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima

from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.utility.compute_coverage import compute_coverage
from src.processing.split_data import train_test_split, split_sequence


def exponential_smoothing(df, cfg):
    train, test = train_test_split(df, cfg['test_size'])
    scaler = MinMaxScaler(feature_range=(10 ** (-10), 1))
    train['y'] = scaler.fit_transform(train.values.reshape(-1, 1))
    test['y'] = scaler.transform(test.values.reshape(-1, 1))

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
            model_es = ExponentialSmoothing(train, seasonal_periods=12,
                                            trend=trend, seasonal=season,
                                            damped=damped)
            model_es = model_es.fit(optimized=True)
            if model_es.aicc < best_aicc:
                best_model_parameters = [trend, season, damped]
                best_aicc = model_es.aicc
    model_es = ExponentialSmoothing(train, seasonal_periods=12,
                                    trend=best_model_parameters[0], seasonal=best_model_parameters[1],
                                    damped=best_model_parameters[2])
    model_es = model_es.fit(optimized=True)
    print(model_es.params)
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
    train, test = train_test_split(df, cfg['test_size'])
    scaler = MinMaxScaler(feature_range=(10 ** (-10), 1))
    train['y'] = scaler.fit_transform(train.values.reshape(-1, 1))
    test['y'] = scaler.transform(test.values.reshape(-1, 1))

    auto_model = auto_arima(train, start_p=1, start_q=1, max_p=11, max_q=11, max_d=3, max_P=5, max_Q=5, max_D=3,
                            m=12, start_P=1, start_Q=1, seasonal=True, d=None, D=None, suppress_warnings=True,
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


def sliding_monte_carlo_forecast(train, test, model, cfg, inherent_noise):
    window_length = int(cfg['forecast_horizon'])
    prediction_sequence = np.zeros([len(test)-window_length, cfg['number_of_mc_forward_passes'], window_length, 1])
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

    return mse_sliding, coverage_95_pi, width_95_pi, coverage_80_pi, width_80_pi


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


# walk-forward validation for univariate data
def pipeline(df, cfg):
    train_and_val, test = train_test_split(df, cfg['test_size'])

    scaler = MinMaxScaler()
    train_and_val = scaler.fit_transform(train_and_val.reshape(-1, 1))
    test = scaler.transform(test.reshape(-1, 1))
    print('Length train', len(train_and_val))
    print('Length test', len(test))

    train, val = train_test_split(train_and_val, cfg['validation_size'])
    train_x, train_y = split_sequence(train, cfg)
    val_x, val_y = split_sequence(np.concatenate([train[-cfg['sequence_length']:], val]), cfg)
    print('trainx', len(train_x))
    print('val', len(val))
    # If using an encoder, extract features from training data,
    model = train_model(train_x, train_y, cfg, val_x, val_y)

    # Compute inherent noise on validation set
    history = [x for x in train]
    y_hat = np.zeros([len(val), train_y.shape[1]])
    for i in range(len(val)):
        x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
        y_hat[i] = model.predict(x_input)[0]
        history.append(val[i])

    inherent_noise = mean_squared_error(val[i:], y_hat[:-i or None, 0])
    print('Validation mse: ', inherent_noise)

    # Predict sequence over testing set using Monte Carlo dropout with n forward passes
    mse_sliding, coverage_95_pi, width_95_pi, coverage_80_pi, width_80_pi = sliding_monte_carlo_forecast(train_and_val,
                                                                                                      test, model,
                                                                                                      cfg,
                                                                                                      inherent_noise)

    return mse_sliding, coverage_95_pi, width_95_pi, coverage_80_pi, width_80_pi


def run_multiple_neural_networks(df, cfg):
    n_runs = 10
    coverage_95_pi = np.zeros([n_runs, cfg['forecast_horizon']])
    width_95_pi = np.zeros([n_runs, cfg['forecast_horizon']])
    coverage_80_pi = np.zeros([n_runs, cfg['forecast_horizon']])
    width_80_pi = np.zeros([n_runs, cfg['forecast_horizon']])
    mse = np.zeros([n_runs, cfg['forecast_horizon']])
    for i in range(n_runs):
        mse_sliding_window, coverage_95_pi_sliding_window, width_95_pi_sliding_window, coverage_80_pi_sliding_window,\
            width_80_pi_sliding_window = pipeline(df['y'].values.reshape(-1, 1), cfg)
        coverage_95_pi[i] = coverage_95_pi_sliding_window
        coverage_80_pi[i] = coverage_80_pi_sliding_window
        width_95_pi[i] = width_95_pi_sliding_window
        width_80_pi[i] = width_80_pi_sliding_window
        mse[i] = mse_sliding_window
    mean_mse = np.mean(mse, axis=0)
    mean_coverage_95 = np.mean(coverage_95_pi, axis=0)
    mean_coverage_80 = np.mean(coverage_80_pi, axis=0)
    mean_width_95 = np.mean(width_95_pi, axis=0)
    mean_width_80 = np.mean(width_80_pi, axis=0)
    print('-----------------------------------------------------------')
    print('MSE sliding window', list(mean_mse))
    print('Coverage 95% PI sliding window',  list(mean_coverage_95))
    print('Width 95% PI sliding window',  list(mean_width_95))
    print('Coverage 80% PI sliding window',  list(mean_coverage_80))
    print('Width 80% PI sliding window',  list(mean_width_80))

    print('Average MSE', np.mean(mean_mse))
    print('Average coverage 95% PI', np.mean(mean_coverage_95))
    print('Average width 95% PI', np.mean(mean_width_95))
    print('Average coverage 80% PI', np.mean(mean_coverage_80))
    print('Average width 80% PI', np.mean(mean_width_80))


def main():
    df, cfg = load_data()

    run_multiple_neural_networks(df, cfg)
    exponential_smoothing(df, cfg)
    arima(df, cfg)
    print(cfg)


if __name__ == '__main__':
    main()
