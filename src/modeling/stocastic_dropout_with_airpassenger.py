import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_autoencoder
from src.dataclasses.Avocado import Avocado
from src.networks.Arima import Sarimax
from src.dataclasses.Airpassengers import Airpassengers
from src.networks.Autoencoder_class import Autoencoder
from src.utility.error_computation import statistics
from src.networks.MonteCarloNetwork import MonteCarloNetwork


def plot_airpassengers(data, pred, mse, pred_es, pred_arima):

    # Compute mean, uncertainty and noise of Monte Carlo forecasts
    mean = np.mean(np.hstack(pred).T, axis=0)
    mc_uncertainty = np.std(np.hstack(pred).T, axis=0)
    inherent_noise = np.sqrt(np.mean(mse))
    uncertainty = np.sqrt(inherent_noise ** 2 + mc_uncertainty ** 2)
    print(len(data))
    plt.plot(np.linspace(1, len(data), len(data)), data, color='green', label='True')

    plt.plot(np.linspace(len(data)-len(pred_es), len(data), len(pred_es)), pred_es[:, 0], color='blue', label='Exponential smoothing')
    plt.plot(np.linspace(len(data)-len(pred_arima), len(data), len(pred_arima)), pred_arima[:, 0], color='red', label='SARIMAX')

    # Plot Neural network with uncertainty
    plt.plot(np.linspace(len(data)-len(mean), len(data), len(mean)), mean, color='orange', label='Neural Network')
    plt.title('Airpassengers data set')
    plt.fill_between(np.linspace(len(data)-len(mean), len(data), len(mean)),
                     mean - 1.28 * uncertainty,
                     mean + 1.28 * uncertainty,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(np.linspace(len(data)-len(mean), len(data), len(mean)),
                     mean - 1.96 * uncertainty,
                     mean + 1.96 * uncertainty,
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.xlabel('Weeks')
    plt.ylabel('')
    plt.legend()
    plt.show()


def pre_training(data, cfg):
    # Train autoencoder as pre training
    train, test = data.get_x()
    encoder, decoder, cfg = build_autoencoder(train, cfg, weights='weights//pretrained_encoder.hdf5')

    # Test autoencoder on holdout data
    predictions = decoder.predict(test)

    mse = 0
    for i in range(len(predictions)):
        mse += mean_squared_error(test[i], predictions[i])
    print('Test mean mse:', mse/len(predictions))

    return encoder, cfg


def stochastic_dropout(x, model, scaler, r, cfg):
    NN = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    trans_input = scaler.transform(x.reshape(-1, 1)).reshape(-1, cfg['sequence_length'], 1)
    NN_pred = NN([trans_input, r])

    return np.vstack(NN_pred)


def stochastic_dropout_with_encoder(x, model, autoencoder, scaler, r, cfg):
    enc = K.function([autoencoder.encoder.layers[0].input, K.learning_phase()], [autoencoder.encoder.layers[-1].output])
    NN = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    enc_pred = np.vstack(enc([x, r]))
    trans_pred = scaler.transform(enc_pred.reshape(-1, autoencoder.encoder_output_dim))\
        .reshape(-1, cfg['sequence_length'], autoencoder.encoder_output_dim)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred)


def monte_carlo_dropout_old(x, y, model, autoencoder, scaler, r, cfg, num_forward_passes):
    mse, predictions = [], []
    for _ in tqdm.tqdm(range(0, num_forward_passes)):
        if autoencoder:
            pred = stochastic_dropout_with_encoder(x, model, autoencoder, scaler, r, cfg)
        else:
            pred = stochastic_dropout(x, model, scaler, r, cfg,)
        mse.append(mean_squared_error(y, pred))
        predictions.append(pred)
    print('Mean mse:', np.mean(mse), 'and std mse:', np.std(mse))
    return mse, predictions


def monte_carlo_dropout(model, x, y, f=None):
    mse, predictions = [], []
    for _ in tqdm.tqdm(range(0, model.num_forward_passes)):
        pred = model.stochastic_dropout(x, f)
        mse.append(mean_squared_error(y, pred))
        predictions.append(pred)
    print('Mean mse:', np.mean(mse), 'and std mse:', np.std(mse))
    return mse, predictions


def pipeline(data, cfg):
    if cfg['autoencoder']:
        # encoder, cfg = pre_training(data=avocado_data, cfg=cfg)
        autoencoder = Autoencoder(data, cfg)
        autoencoder.train()
        autoencoder.test()
    else:
        autoencoder = None
    # Extract data
    train_x, train_y, train_f = data.get_train_sequence()
    plt.figure()
    plt.plot(data.data)
    plt.show()

    # Fit model
    mc_model = MonteCarloNetwork(data, autoencoder, cfg)
    mc_model.train(train_x, train_y, train_f)

    # model = train_model(train_x, train_y, cfg)

    # Fit seasonal arima
    # sarimax = Sarimax(data.get, cfg)

    test_x, test_y, test_f = data.get_test_sequence()

    # Forecast on the last proportion of the data set
    mse_test, pred_test = monte_carlo_dropout(mc_model, test_x, test_y, test_f)

    model_es = ExponentialSmoothing(data.train)
    model_es = model_es.fit()
    pred_es = model_es.predict(start=data.test.index[cfg['sequence_length']],
                               end=data.test.index[-1])
    model_arima = SARIMAX(data.train, order=(3, 1, 0))
    model_arima = model_arima.fit()
    pred_arima = model_arima.predict(start=data.test.index[cfg['sequence_length']],
                                     end=data.test.index[-1])

    pred_es = np.asarray(pred_es).reshape(test_y.shape)
    pred_arima = np.asarray(pred_arima).reshape(test_y.shape)
    print('======= Test Statistics =======')
    statistics(test_x, test_y, mse_test, pred_test)
    print('Exponential Smoothing:', mean_squared_error(test_y, pred_es))
    print('SARIMAX:', mean_squared_error(test_y, pred_arima))
    plt.figure()
    plt.plot(data.data)
    plt.show()
    plot_airpassengers(data.data, pred_test, mse_test, pred_es, pred_arima)


def main():
    df, cfg = load_data()

    data = Airpassengers(cfg)

    pipeline(data, cfg)

    print(cfg)


if __name__ == '__main__':
    main()
