import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras import backend as K

from src.processing.transform_to_stationary import difference_transformation, inverse_difference_transformation
from src.modeling.config.open_config import load_config_file
from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_autoencoder
from src.processing.split_data import split_sequence, train_test_split
from src.processing.avocado_price_data_set import process_avocado_data
from src.utility.compute_coverage import compute_coverage
from src.modeling.pre_traning_autoencoder import pre_training
from src.dataclasses.Avocado import Avocado


def plot_predictions(df, mean, median, quantile_80, quantile_95, cfg):
    x_data = np.linspace(1, len(df), len(df))
    for i in range(cfg['forecasting_horizon']):
        x_predictions = np.linspace(len(df) - len(mean[:, i]) + 1+i, len(df)+i, len(mean[:, i]))
        plt.figure()
        plt.title("Time Series Forecasting Multi-step")
        plt.plot(x_data, df, label='Data')
        plt.plot(x_predictions, mean[:, i], label='Mean')
        plt.plot(x_predictions, median[:, i], label='Median')
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


def monte_carlo_forecast(train, test, model, cfg, encoder=None):
    prediction_sequence = np.zeros([cfg['number_of_mc_forward_passes'], len(test), cfg['forecasting_horizon']])
    if encoder:
        enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    # Number of MC samples
    print("=== Forwarding", cfg['number_of_mc_forward_passes'], "passes ===")
    diff = 0
    for j in tqdm.tqdm(range(cfg['number_of_mc_forward_passes'])):
        history = train
        # Prediction horizon / test length
        for i in range(len(test)):
            # fit model and make forecast for history
            x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], cfg['num_features']))
            if encoder:
                x_input_enc = np.array(history[cfg['target_feature']].iloc[-cfg['sequence_length']:]).reshape(
                    (1, cfg['sequence_length'], 1))
                features_last_step = history.iloc[-1].values
                features_last_step = features_last_step.reshape(1, features_last_step.shape[0], 1)
                enc_pred = np.vstack(enc([x_input_enc, cfg['dropout_rate_test']]))
                enc_pred = np.concatenate([enc_pred, features_last_step], axis=1)
                x_input = enc_pred
            mc_sample = func([x_input, cfg['dropout_rate_test']])[0]
            # store forecast in list of predictions
            history.loc[len(history)] = test.iloc[i]
            if cfg['multi_step_prediction']:
                history.loc[len(history)-1, cfg['target_feature']] = mc_sample[0, 0]
            prediction_sequence[j, i] = mc_sample
            diff += prediction_sequence[j, i]
    diff = diff/(cfg['number_of_mc_forward_passes']*len(test))
    print(diff)
    return prediction_sequence


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def train_autoencoder(data, cfg):
    # Train autoencoder as pre training
    encoder_train = np.concatenate([data.train_conv, data.train_org], axis=0)
    encoder, decoder, cfg = build_autoencoder(encoder_train, cfg, weights='weights//pretrained_encoder.hdf5')
    return encoder, cfg


# walk-forward validation for univariate data
def walk_forward_validation(data, cfg, encoder=None):
    # If using an encoder, extract features from training data,
    if encoder:
        enc_pred_train = encoder.predict(data.x_train)
        train_x = np.concatenate([enc_pred_train, data.f_train], axis=2)
        enc_pred_test = encoder.predict(data.x_test)
        test_x = np.concatenate([enc_pred_test, data.f_test], axis=2)
        scaler1 = StandardScaler()
        train_x = scaler1.fit_transform(train_x.reshape(-1, 128 + 8)).reshape(-1, cfg['sequence_length'], 128 + 8)
        test_x = scaler1.transform(test_x.reshape(-1, 128 + 8)).reshape(-1, cfg['sequence_length'], 128 + 8)
    model = train_model(train_x, data.y_train, cfg)

    # Compute inherent noise on validation set
    history = train
    y_hat = np.zeros([len(val), train_y.shape[1]])
    for i in range(len(val)):
        x_input = np.array(history[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], cfg['num_features']))
        if encoder:
            x_input_enc = np.array(history[cfg['target_feature']].iloc[-cfg['sequence_length']:]).reshape((1, cfg['sequence_length'], 1))
            enc_pred = encoder.predict(x_input_enc)
            features_last_step = history.iloc[-1].values
            features_last_step = features_last_step.reshape(1, features_last_step.shape[0], 1)
            # train_feature = np.concatenate([enc_pred, features_last_step], axis=1)
            x_input = np.concatenate([enc_pred, features_last_step], axis=1)
        y_hat[i] = model.predict(x_input)[0]
        history.append(val.iloc[i])
    inherent_noise = np.zeros(cfg['forecasting_horizon'])
    for i in range(cfg['forecasting_horizon']):
        inherent_noise[i] = measure_rmse(val[cfg['target_feature']].iloc[i:], y_hat[:-i or None, 0])

    # Predict sequence over testing set using Monte Carlo dropout with n forward passes
    prediction_sequence = monte_carlo_forecast(train_and_val, test, model, cfg, encoder)

    # Compute mean and uncertainty for the Monte Carlo estimates
    mc_mean = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    mc_median = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    mc_uncertainty = np.zeros([prediction_sequence.shape[1], prediction_sequence.shape[2]])
    for i in range(cfg['forecasting_horizon']):
        mc_mean[:, i] = prediction_sequence[:, :, i].mean(axis=0)
        mc_median[:, i] = np.median(prediction_sequence[:, :, i], axis=0)
        mc_uncertainty[:, i] = prediction_sequence[:, :, i].std(axis=0)
    # Add inherent noise and uncertainty obtained from Monte Carlo samples
    total_uncertainty = np.sqrt(inherent_noise**2 + mc_uncertainty**2)
    # estimate prediction error
    error = measure_rmse(test[cfg['target_feature']], mc_mean)
    print(' > %.3f' % error)

    # Compute quantiles of the Monte Carlo estimates
    quantile_80 = [np.quantile(prediction_sequence[:, :, 0], 0.10, axis=0), np.quantile(prediction_sequence[:, :, 0], 0.90, axis=0)]
    quantile_95 = [np.quantile(prediction_sequence[:, :, 0], 0.025, axis=0), np.quantile(prediction_sequence[:, :, 0], 0.975, axis=0)]
    """
    coverage_80pi = compute_coverage(upper_limits=quantile_80[1],
                                     lower_limits=quantile_80[0],
                                     actual_values=test['AveragePrice'])
    coverage_95pi = compute_coverage(upper_limits=quantile_95[1],
                                     lower_limits=quantile_95[0],
                                     actual_values=test['AveragePrice'])
    # print('80%-prediction interval coverage: ', coverage_80pi)
    # print('95%-prediction interval coverage: ', coverage_95pi)
    """
    for i in range(cfg['forecasting_horizon']):
        coverage_80pi = compute_coverage(upper_limits=mc_mean[:, i]+1.28*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.28*total_uncertainty[:, i],
                                         actual_values=test[[cfg['target_feature']]].values)
        coverage_95pi = compute_coverage(upper_limits=mc_mean[:, i]+1.96*total_uncertainty[:, i],
                                         lower_limits=mc_mean[:, i]-1.96*total_uncertainty[:, i],
                                         actual_values=test[[cfg['target_feature']]].values)
        print('80%-prediction interval coverage: ', i, coverage_80pi)
        print('95%-prediction interval coverage: ', i, coverage_95pi)

    return prediction_sequence, mc_mean, mc_median, total_uncertainty, test[[cfg['target_feature']]]


def stochastic_dropout(x, f, model, encoder, scaler, r, cfg):
    enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    NN = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    enc_pred = np.vstack(enc([x, r]))
    enc_pred = np.concatenate([enc_pred, f], axis=2)
    trans_pred = scaler.transform(enc_pred.reshape(-1, 128 + 8)).reshape(-1, cfg['sequence_length'], 128 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred)


def walk_forward_validation_2(data, cfg, encoder=None):
    # If using an encoder, extract features from training data,
    enc_pred_train = encoder.predict(data.x_train)
    train_x = np.concatenate([enc_pred_train, data.f_train], axis=2)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(-1, 128 + 8)).reshape(-1, cfg['sequence_length'], 128 + 8)
    model = train_model(train_x, data.y_train, cfg)

    test_scores = []
    for _ in tqdm.tqdm(range(0, 200)):
        pred = stochastic_dropout(data.x_test, data.f_test, model, encoder, scaler, 0.5, cfg)
        test_scores.append(mean_absolute_error(pred, data.y_test))
    print(np.mean(test_scores), np.std(test_scores))

    mae_holdout = []
    pred_holdout = []
    for _ in tqdm.tqdm(range(0, 200)):
        pred = stochastic_dropout(data.x_holdout_org, data.f_holdout_org, model, encoder, scaler, 0.5, cfg)
        mae_holdout.append(mean_absolute_error(pred, data.y_holdout_org))
        pred_holdout.append(pred)
    print(np.mean(mae_holdout), np.std(mae_holdout))

    uncertainty = np.sqrt(np.mean(test_scores) + np.std(np.hstack(pred_holdout).T, axis=0)**2)
    plt.plot(np.mean(np.hstack(pred_holdout).T, axis=0), color='orange')
    plt.plot(data.y_holdout_org, color='green')
    plt.title('AveragePrice CONVENTIONAL Albany - Model1')
    plt.fill_between(np.linspace(1, len(data.y_holdout_org), len(data.y_holdout_org)),
                     np.mean(np.hstack(pred_holdout).T, axis=0) - 1.28*uncertainty,
                     np.mean(np.hstack(pred_holdout).T, axis=0) + 1.28*uncertainty,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(np.linspace(1, len(data.y_holdout_org), len(data.y_holdout_org)),
                     np.mean(np.hstack(pred_holdout).T, axis=0) - 1.96*uncertainty,
                     np.mean(np.hstack(pred_holdout).T, axis=0) + 1.96*uncertainty,
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()


def main():
    cfg = load_config_file('config\\config.yml', print_config=True)

    df = load_data(cfg)
    #if cfg['differencing']:
    #    df = df.diff(periodes=1)
    print(df.shape)
    cfg['target_feature'] = 'AveragePrice'
    cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
    df[cols] = df[cols].diff(periods=1, axis=1)
    df = df.drop(df.index[0])
    avocado_data = Avocado(df, cfg)
    scaler = StandardScaler()
    # df_train, cfg = process_avocado_data(df_train, cfg)
    # df_test, cfg = process_avocado_data(df_test, cfg)
    # df_train[df_train.columns.values] = scaler.fit_transform(df_train[df_train.columns.values])
    # df_test[df_test.columns.values] = scaler.transform(df_test[df_test.columns.values])

    if cfg['autoencoder']:
        encoder, cfg = pre_training(data=avocado_data, cfg=cfg)
    else:
        encoder = None
    #predictions, mc_mean, mc_median, total_uncertainty, _ = walk_forward_validation(avocado_data, cfg, encoder)
    walk_forward_validation_2(avocado_data, cfg, encoder)
    #plot_predictions(df.loc[:, pd.IndexSlice['AveragePrice', 'Albany', 'conventional']],
    #                 mc_mean, mc_median,
    #                 np.array([mc_mean - 1.28*total_uncertainty, mc_mean + 1.28*total_uncertainty]),
    #                 np.array([mc_mean - 1.96*total_uncertainty, mc_mean + 1.96*total_uncertainty]), cfg)
    print(cfg)


if __name__ == '__main__':
    main()
