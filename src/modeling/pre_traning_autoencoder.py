import numpy as np
from sklearn.metrics import mean_squared_error

from src.processing.split_data import split_sequence
from src.networks.autoencoder import build_autoencoder


def pre_training(df, cfg):
    # Train autoencoder as pre training
    encoder_train = df[:-int(len(df) * cfg['test_size'])]
    print(encoder_train)
    train_x = []
    for name, values in df.iteritems():
        x, _ = split_sequence(values.values.reshape(-1, 1), cfg)
        train_x.append(x)
    train_x = np.asarray(train_x)
    train_x = train_x.reshape([train_x.shape[0]*train_x.shape[1], train_x.shape[2], train_x.shape[3]])
    print(train_x.shape)
    encoder, decoder, cfg = build_autoencoder(train_x, cfg, weights='weights//pretrained_encoder.hdf5')

    # Test autoencoder on holdout data
    #encoder_test = data.x_test
    #predictions = decoder.predict(encoder_test)

    #mse = 0
    #for i in range(len(predictions)):
    #    mse += mean_squared_error(encoder_test[i], predictions[i])
    #print('Test mean mse:', mse/len(predictions))

    return encoder, cfg
