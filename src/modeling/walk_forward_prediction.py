import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.processing.scale import scale_data
from src.processing.transform_to_stationary import difference_transformation, inverse_difference_transformation
from src.modeling.config.open_config import load_config_file
from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from keras import backend as K


def plot_predictions(df, mean, quantile_80, quantile_95, cfg):
    x_data = np.linspace(1, len(df), len(df))
    for i in range(cfg['forecasting_horizon']):
        x_predictions = np.linspace(len(df) - len(mean[:, i]) + 1+i, len(df)+i, len(mean[:, i]))
        plt.figure()
        plt.title("Time Series Forecasting Multi-step")
        plt.plot(x_data, df, label='Data')
        plt.plot(x_predictions, mean[:, i], label='Predictions')
        plt.fill_between(x_predictions, quantile_80[0, :, i], quantile_80[1, :, i],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_predictions, quantile_95[0, :, i], quantile_95[1, :, i],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()


def train_test_split(df, test_split=0.2):
    test_size = int(test_split*len(df))
    return df[:-test_size], df[-test_size:]


# split a univariate sequence into samples
def split_sequence(sequence, cfg):
    x, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + cfg['sequence_length']
        out_end_ix = end_ix + cfg['forecasting_horizon']
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


# forecast with a pre-fit model
def forecast(model, history, cfg, use_dropout=True):
    # MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    # prepare data
    x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))

    # forecast
    monte_carlo_samples = model.predict(x_input)
    # MC_samples = MC_output([x_input, use_dropout])[0]

    return monte_carlo_samples[0]


def monte_carlo_forecast(train_and_val, test, model, cfg):
    prediction_sequence = np.zeros([cfg['number_of_mc_forward_passes'], len(test), cfg['forecasting_horizon']])
    func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    # Number of MC samples
    for j in range(cfg['number_of_mc_forward_passes']):
        history = [x for x in train_and_val]
        # Prediction horizon / test length
        for i in range(len(test)):
            # fit model and make forecast for history
            x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
            # mc_sample = model.predict(x_input)[0]
            mc_sample = func([x_input, 1])[0]
            # store forecast in list of predictions
            if cfg['multi_step_prediction']:
                history.append(mc_sample[0, 0])
            else:
                history.append(test[i])
            prediction_sequence[j, i] = mc_sample
    return prediction_sequence


def compute_coverage(upper_limits, lower_limits, actual_values):
    coverage = 0
    for i in range(len(actual_values)):
        if lower_limits[i] < actual_values[i] < upper_limits[i]:
            coverage += 1
    return coverage/len(actual_values)


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


# walk-forward validation for univariate data
def walk_forward_validation(df, cfg):
    # Split data in train, val and test set
    train_and_val, test = train_test_split(df, test_split=cfg['test_size'])
    train, val = train_test_split(train_and_val, test_split=0.1)

    # Split training data in sequences
    train_x, train_y = split_sequence(train, cfg)
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1]))
    model = train_model(train_x, train_y, cfg)

    # Compute inherent noise on validation set
    history = [x for x in train]
    y_hat = np.zeros([len(val), train_y.shape[1]])
    for i in range(len(val)):
        x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
        y_hat[i] = model.predict(x_input)[0]
        history.append(val[i])
    inherent_noise = np.zeros(cfg['forecasting_horizon'])
    for i in range(cfg['forecasting_horizon']):
        inherent_noise[i] = measure_rmse(val[i:], y_hat[:-i or None, 0])

    prediction_sequence = monte_carlo_forecast(train_and_val, test, model, cfg)
    mc_mean = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    mc_uncertainty = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    for i in range(cfg['forecasting_horizon']):
        mc_mean[:, i] = prediction_sequence[:, :, i].mean(axis=0)
        mc_uncertainty[:, i] = prediction_sequence[:, :, i].std(axis=0)
    total_uncertainty = np.sqrt(inherent_noise**2 + mc_uncertainty**2)

    # estimate prediction error
    # error = measure_rmse(test, mc_mean)
    # print(' > %.3f' % error)
    quantile_80 = [np.quantile(prediction_sequence[:, :, 0], 0.10, axis=0), np.quantile(prediction_sequence[:, :, 0], 0.90, axis=0)]
    quantile_95 = [np.quantile(prediction_sequence[:, :, 0], 0.025, axis=0), np.quantile(prediction_sequence[:, :, 0], 0.975, axis=0)]
    coverage_80pi = compute_coverage(upper_limits=quantile_80[1],
                                     lower_limits=quantile_80[0],
                                     actual_values=test)
    coverage_95pi = compute_coverage(upper_limits=quantile_95[1],
                                     lower_limits=quantile_95[0],
                                     actual_values=test)
    #print('80%-prediction interval coverage: ', coverage_80pi)
    #print('95%-prediction interval coverage: ', coverage_95pi)
    for i in range(cfg['forecasting_horizon']):
        coverage_80pi = compute_coverage(upper_limits=mc_mean[:, i]+1.28*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.28*total_uncertainty[:, i],
                                         actual_values=test)
        coverage_95pi = compute_coverage(upper_limits=mc_mean[:, i]+1.96*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.96*total_uncertainty[:, i],
                                         actual_values=test)
        # print('80%-prediction interval coverage: ', i, coverage_80pi)
        # print('95%-prediction interval coverage: ', i, coverage_95pi)
    return prediction_sequence, mc_mean, total_uncertainty, quantile_80, quantile_95, test


def main():
    cfg = load_config_file('config\\config.yml', print_config=True)

    df = load_data(cfg)
    df = scale_data(df)

    if cfg['differencing']:
        df_difference = difference_transformation(df)
        predictions_difference, mc_mean_difference, total_uncertainty_difference, difference_quantile_80, difference_quantile_95, _ = walk_forward_validation(df_difference, cfg)
        predictions = inverse_difference_transformation(df, predictions_difference)
        mc_mean = inverse_difference_transformation(df, mc_mean_difference)
        total_uncertainty = inverse_difference_transformation(df, total_uncertainty_difference)
        quantile_80 = [inverse_difference_transformation(df, difference_quantile_80[0]), inverse_difference_transformation(df, difference_quantile_80[1])]
        quantile_95 = [inverse_difference_transformation(df, difference_quantile_95[0]), inverse_difference_transformation(df, difference_quantile_95[1])]

    else:
        predictions, mc_mean, total_uncertainty, quantile_80, quantile_95, _ = walk_forward_validation(df, cfg)

    # predictions, mc_mean, total_uncertainty, quantile_80, quantile_95 = walk_forward_validation(df, cfg)

    # plot_predictions(df, mc_mean, quantile_80, quantile_95, cfg)
    plot_predictions(df, mc_mean, np.array([mc_mean - 1.28*total_uncertainty, mc_mean+1.28*total_uncertainty]),
                     np.array([mc_mean - 1.96*total_uncertainty, mc_mean + 1.96*total_uncertainty]), cfg)


if __name__ == '__main__':
    main()
