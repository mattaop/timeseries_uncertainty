from keras import Model
from keras.layers import *


def build_model(train_x, cfg):
    inp = Input(shape=(cfg['sequence_length'], 1))
    encoded_ae = LSTM(cfg['n_feature_extraction'], activation='relu', return_sequences=True)(inp)
    encoded_ae = Dropout(0.3)(encoded_ae)
    decoded_ae = LSTM(32, activation='relu', return_sequences=True)(encoded_ae)
    decoded_ae = Dropout(0.3)(decoded_ae)
    out_ae = TimeDistributed(Dense(1))(decoded_ae)
    sequence_autoencoder = Model(inp, out_ae, name='autoencoder')
    encoder = Model(inp, encoded_ae, name='encoder')

    sequence_autoencoder.compile(optimizer='adam', loss='mse')
    sequence_autoencoder.fit(train_x, train_x, epochs=100, batch_size=16, verbose=2, shuffle=True)
    return encoder
