import numpy as np
from sklearn.metrics import mean_squared_error

from src.processing.split_data import split_sequence
from src.networks.autoencoder import build_autoencoder


def pre_training(data, cfg):
    # Train autoencoder as pre training
    encoder_train = data.x_train
    encoder, decoder, cfg = build_autoencoder(encoder_train, cfg, weights='weights//pretrained_encoder.hdf5')

    # Test autoencoder on holdout data
    encoder_test = data.x_test
    predictions = decoder.predict(encoder_test)

    mse = 0
    for i in range(len(predictions)):
        mse += mean_squared_error(encoder_test[i], predictions[i])
    print('Test mean mse:', mse/len(predictions))

    return encoder, cfg
