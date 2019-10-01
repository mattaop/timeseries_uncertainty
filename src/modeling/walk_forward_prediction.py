import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

from src.processing.transform_to_stationary import difference_transformation, inverse_difference_transformation
from src.modeling.config.open_config import load_config_file
from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_model
from src.processing.split_data import split_sequence, train_test_split
from src.utility.compute_coverage import compute_coverage


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


# forecast with a pre-fit model
def forecast(model, history, cfg):
    # prepare data
    x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))

    # forecast
    monte_carlo_samples = model.predict(x_input)
    return monte_carlo_samples[0]


def monte_carlo_forecast(train_and_val, test, model, scaler, cfg, encoder=None):
    prediction_sequence = np.zeros([cfg['number_of_mc_forward_passes'], len(test), cfg['forecasting_horizon']])
    if encoder:
        enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    # Number of MC samples
    print("=== Forwarding", cfg['number_of_mc_forward_passes'], "passes ===")
    for j in tqdm.tqdm(range(cfg['number_of_mc_forward_passes'])):
        history = [x for x in train_and_val]
        # Prediction horizon / test length
        for i in range(len(test)):
            # fit model and make forecast for history
            x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))

            if encoder:
                enc_pred = np.vstack(enc([x_input, cfg['dropout_rate_test']]))
                enc_pred = np.concatenate([x_input, enc_pred], axis=2)
                #enc_pred = scaler.transform(enc_pred.reshape(-1, cfg['n_feature_extraction']))\
                #    .reshape(-1, cfg['sequence_length'], cfg['n_feature_extraction'])
                x_input = enc_pred

            mc_sample = func([x_input, cfg['dropout_rate_test']])[0]
            # store forecast in list of predictions
            if cfg['multi_step_prediction']:
                history.append(mc_sample[0, 0])
            else:
                history.append(test[i])
            prediction_sequence[j, i] = mc_sample
    return prediction_sequence


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def train_autoencoder(df, cfg):
    train, _ = train_test_split(df, test_split=cfg['test_size'])
    train_x, _ = split_sequence(train, cfg)
    encoder = build_model(train_x, cfg)
    return encoder


# walk-forward validation for univariate data
def walk_forward_validation(df, scaler, cfg, encoder=None):
    # Split data in train, val and test set
    train_and_val, test = train_test_split(df, test_split=cfg['test_size'])
    train, val = train_test_split(train_and_val, test_split=0.1)

    # Split training data in sequences
    train_x, train_y = split_sequence(train, cfg)

    # If using an encoder, extract features from training data,
    if encoder:
        enc_pred = encoder.predict(train_x)
        print(train_x.shape)
        print(enc_pred.shape)
        train_x = np.concatenate([train_x, enc_pred], axis=2)
    model = train_model(train_x, train_y, cfg)

    # Compute inherent noise on validation set
    history = [x for x in train]
    y_hat = np.zeros([len(val), train_y.shape[1]])
    for i in range(len(val)):
        x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
        if encoder:
            enc_pred = encoder.predict(x_input)
            print(enc_pred.shape)
            print(x_input.shape)
            x_input = np.concatenate([x_input, enc_pred], axis=2)
        y_hat[i] = model.predict(x_input)[0]
        history.append(val[i])
    inherent_noise = np.zeros(cfg['forecasting_horizon'])
    for i in range(cfg['forecasting_horizon']):
        inherent_noise[i] = measure_rmse(val[i:], y_hat[:-i or None, 0])

    # Predict sequence over testing set using Monte Carlo dropout with n forward passes
    prediction_sequence = monte_carlo_forecast(train_and_val, test, model, scaler, cfg, encoder)

    # Compute mean and uncertainty for the Monte Carlo estimates
    mc_mean = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    mc_uncertainty = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    for i in range(cfg['forecasting_horizon']):
        mc_mean[:, i] = prediction_sequence[:, :, i].mean(axis=0)
        mc_uncertainty[:, i] = prediction_sequence[:, :, i].std(axis=0)

    # Add inherent noise and uncertainty obtained from Monte Carlo samples
    total_uncertainty = np.sqrt(inherent_noise**2 + mc_uncertainty**2)

    # estimate prediction error
    error = measure_rmse(test, mc_mean)
    print(' > %.3f' % error)

    # Compute quantiles of the Monte Carlo estimates
    quantile_80 = [np.quantile(prediction_sequence[:, :, 0], 0.10, axis=0), np.quantile(prediction_sequence[:, :, 0], 0.90, axis=0)]
    quantile_95 = [np.quantile(prediction_sequence[:, :, 0], 0.025, axis=0), np.quantile(prediction_sequence[:, :, 0], 0.975, axis=0)]
    coverage_80pi = compute_coverage(upper_limits=quantile_80[1],
                                     lower_limits=quantile_80[0],
                                     actual_values=test)
    coverage_95pi = compute_coverage(upper_limits=quantile_95[1],
                                     lower_limits=quantile_95[0],
                                     actual_values=test)
    # print('80%-prediction interval coverage: ', coverage_80pi)
    # print('95%-prediction interval coverage: ', coverage_95pi)

    for i in range(cfg['forecasting_horizon']):
        coverage_80pi = compute_coverage(upper_limits=mc_mean[:, i]+1.28*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.28*total_uncertainty[:, i],
                                         actual_values=test)
        coverage_95pi = compute_coverage(upper_limits=mc_mean[:, i]+1.96*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.96*total_uncertainty[:, i],
                                         actual_values=test)
        print('80%-prediction interval coverage: ', i, coverage_80pi)
        print('95%-prediction interval coverage: ', i, coverage_95pi)

    return prediction_sequence, mc_mean, total_uncertainty, quantile_80, quantile_95, test


def main():
    cfg = load_config_file('config\\config.yml', print_config=True)

    df = load_data(cfg)
    scaler = MinMaxScaler(feature_range=[0, 1])
    df = scaler.fit_transform(df)

    if cfg['differencing']:
        df_difference = difference_transformation(df)
        predictions_difference, mc_mean_difference, total_uncertainty_difference, difference_quantile_80, difference_quantile_95, _ = walk_forward_validation(df_difference, cfg)
        predictions = inverse_difference_transformation(df, predictions_difference)
        mc_mean = inverse_difference_transformation(df, mc_mean_difference)
        total_uncertainty = inverse_difference_transformation(df, total_uncertainty_difference)
        quantile_80 = [inverse_difference_transformation(df, difference_quantile_80[0]), inverse_difference_transformation(df, difference_quantile_80[1])]
        quantile_95 = [inverse_difference_transformation(df, difference_quantile_95[0]), inverse_difference_transformation(df, difference_quantile_95[1])]

    else:
        encoder = train_autoencoder(df, cfg)
        predictions, mc_mean, total_uncertainty, quantile_80, quantile_95, _ = walk_forward_validation(df, scaler, cfg, encoder)

    # plot_predictions(df, mc_mean, quantile_80, quantile_95, cfg)
    plot_predictions(df, mc_mean, np.array([mc_mean - 1.28*total_uncertainty, mc_mean+1.28*total_uncertainty]),
                     np.array([mc_mean - 1.96*total_uncertainty, mc_mean + 1.96*total_uncertainty]), cfg)


if __name__ == '__main__':
    main()
