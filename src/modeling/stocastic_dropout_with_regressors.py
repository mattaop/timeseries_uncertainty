import numpy as np
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import backend as K
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.networks.autoencoder import build_autoencoder
from src.dataclasses.Avocado import Avocado
from src.networks.Arima import Sarimax
from src.dataclasses.Airpassengers import Airpassengers
from src.networks.Autoencoder_class import Autoencoder
from src.utility.error_computation import statistics


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


def stochastic_dropout(x, f, model, scaler, r, cfg, num_features):
    NN = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    x_input = np.concatenate([x, f], axis=2)
    trans_input = scaler.transform(x_input.reshape(-1, num_features)).reshape(-1, cfg['sequence_length'], num_features)
    NN_pred = NN([trans_input, r])

    return np.vstack(NN_pred)


def stochastic_dropout_with_encoder(x, f, model, autoencoder, scaler, r, cfg, num_features):
    enc = K.function([autoencoder.encoder.layers[0].input, K.learning_phase()], [autoencoder.encoder.layers[-1].output])
    NN = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    enc_pred = np.vstack(enc([x, r]))
    enc_pred = np.concatenate([enc_pred, f], axis=2)
    trans_pred = scaler.transform(enc_pred.reshape(-1, autoencoder.encoder_output_dim + num_features-1))\
        .reshape(-1, cfg['sequence_length'], autoencoder.encoder_output_dim + num_features-1)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred)


def monte_carlo_dropout(x, f, y, model, autoencoder, scaler, r, cfg, num_features, num_forward_passes):
    mse, predictions = [], []
    for _ in tqdm.tqdm(range(0, num_forward_passes)):
        if autoencoder:
            pred = stochastic_dropout_with_encoder(x, f, model, autoencoder, scaler, r, cfg, num_features)
        else:
            pred = stochastic_dropout(x, f, model, scaler, r, cfg, num_features)
        mse.append(mean_squared_error(y, pred))
        predictions.append(pred)
    print('Mean mse:', np.mean(mse), 'and std mse:', np.std(mse))
    return mse, predictions


def pipeline(data, cfg, autoencoder=None):

    # Extract data
    train_x, train_f, train_y = data.get_train_sequence()

    # Predict with encoder, concatenate and scale data
    scaler = StandardScaler()
    if autoencoder:
        enc_pred_train = autoencoder.encoder.predict(train_x)
        train_x = np.concatenate([enc_pred_train, train_f], axis=2)
        train_x = scaler.fit_transform(train_x.reshape(-1, autoencoder.encoder_output_dim + data.num_features-1))\
            .reshape(-1, cfg['sequence_length'], autoencoder.encoder_output_dim + data.num_features-1)
    else:
        train_x = np.concatenate([train_x, train_f], axis=2)
        train_x = scaler.fit_transform(train_x.reshape(-1, data.num_features)) \
            .reshape(-1, cfg['sequence_length'],  data.num_features)

    # Fit LSTM model
    model = train_model(train_x, train_y, cfg)

    # Fit Exponential Smoothing
    model_es = ExponentialSmoothing(np.asarray(data.train['AveragePrice']))
    model_es = model_es.fit()

    # Fit seasonal arima
    # sarimax = Sarimax(data.get, cfg)

    test_x, test_f, test_y = data.get_test_sequence()

    # Forecast on the last proportion of the data sets
    mse_test, pred_test = monte_carlo_dropout(test_x, test_f, test_y, model, autoencoder, scaler, 0.5, cfg,
                                              data.num_features, num_forward_passes=cfg['number_of_mc_forward_passes'])
    pred_es = model_es.predict(test_x)

    print('======= Test Statistics =======')
    statistics(test_x, test_y, mse_test, pred_test, pred_es)

    # Extract hold-out data set
    holdout_x, holdout_f, holdout_y = data.get_holdout_sequence(avocado_types=['organic'])

    # Forecast on unseen hold-out set
    mse_holdout, pred_holdout = monte_carlo_dropout(holdout_x, holdout_f, holdout_y, model, autoencoder, scaler, 0.5,
                                                    cfg, data.num_features,
                                                    num_forward_passes=cfg['number_of_mc_forward_passes'])
    print('======= Holdout Statistics =======')
    statistics(holdout_x, holdout_y, mse_holdout, pred_holdout)


def main():
    df, cfg = load_data()
    # if cfg['differencing']:
    #    df = df.diff(periodes=1)
    print(df.shape)
    cfg['target_feature'] = 'AveragePrice'
    cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
    data = Avocado(cfg)
    # data.data[data.data.columns.values] = data.data[data.data.columns.values].diff(periods=1, axis=0)
    # data.data = data.data.drop(data.data.index[0])

    # data = Airpassengers(cfg)
    if cfg['autoencoder']:
        # encoder, cfg = pre_training(data=avocado_data, cfg=cfg)
        autoencoder = Autoencoder(data, cfg)
        autoencoder.train()
        autoencoder.test()
        pipeline(data, cfg, autoencoder)
    else:
        pipeline(data, cfg)

    print(cfg)


if __name__ == '__main__':
    main()
