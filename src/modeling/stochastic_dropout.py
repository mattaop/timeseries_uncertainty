import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    # Fit model
    mc_model = MonteCarloNetwork(data, autoencoder, cfg)
    mc_model.train(train_x, train_y, train_f)

    # model = train_model(train_x, train_y, cfg)

    # Fit seasonal arima
    # sarimax = Sarimax(data.get, cfg)

    test_x, test_y, test_f = data.get_test_sequence()

    # Forecast on the last proportion of the data sets
    # mse_test, pred_test = monte_carlo_dropout(test_x, test_y, model, autoencoder, scaler, 0.5, cfg,
    #                                          num_forward_passes=cfg['number_of_mc_forward_passes']
    mse_test, pred_test = monte_carlo_dropout(mc_model, test_x, test_y, test_f)
    """
    pred_es = []
    pred_arima = []
    for region in data.regions:
        if region not in data.holdout_region:
            for avocado_type in data.avocado_types:
                model_es = ExponentialSmoothing(data.train.loc[:, pd.IndexSlice['AveragePrice', region, avocado_type]],
                                                seasonal_periods=52)
                model_es = model_es.fit()
                pred_es.append(model_es.predict(start=data.test.index[cfg['sequence_length']],
                                                end=data.test.index[-1]))
                model_arima = SARIMAX(data.train.loc[:, pd.IndexSlice['AveragePrice', region, avocado_type]], order=(3, 1, 0))
                model_arima = model_arima.fit()
                pred_arima.append(model_arima.predict(start=data.test.index[cfg['sequence_length']],
                                                      end=data.test.index[-1]))

    pred_es = np.asarray(pred_es).reshape(test_y.shape)
    pred_arima = np.asarray(pred_arima).reshape(test_y.shape)
    print('Exponential Smoothing:', mean_squared_error(test_y, pred_es))
    print('SARIMAX:', mean_squared_error(test_y, pred_arima))
    """
    print('======= Test Statistics =======')
    statistics(test_x, test_y, mse_test, pred_test)

    # Extract hold-out data set
    holdout_x, holdout_y, holdout_f = data.get_holdout_sequence(avocado_types=['organic'])

    # Forecast on unseen hold-out set
    # mse_holdout, pred_holdout = monte_carlo_dropout(holdout_x, holdout_y, model, autoencoder, scaler, 0.5, cfg,
    #                                                 num_forward_passes=cfg['number_of_mc_forward_passes'])
    mse_holdout, pred_holdout = monte_carlo_dropout(mc_model, holdout_x, holdout_y, holdout_f)

    print('======= Holdout Statistics =======')
    statistics(holdout_x, holdout_y, mse_holdout, pred_holdout)


def main():
    df, cfg = load_data()

    data = Avocado(cfg)

    pipeline(data, cfg)

    print(cfg)


if __name__ == '__main__':
    main()
