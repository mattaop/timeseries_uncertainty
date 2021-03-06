from keras import Model
from keras.layers import *
from keras_radam import RAdam
from keras_lookahead import Lookahead


def build_autoencoder(train, cfg, weights=None):
    if cfg['autoencoder'].lower() == 'lstm':
        encoder, decoder, cfg = build_lstm_encoder(train, cfg, weights)
    elif cfg['autoencoder'].lower() == 'cnn':
        encoder, decoder, cfg = build_cnn_encoder(train, cfg, weights)
    elif cfg['autoencoder'].lower() == 'rnn':
        encoder, decoder, cfg = build_rnn_encoder(train, cfg, weights)
    else:
        encoder = None
        decoder = None
        cfg = None
    return encoder, decoder, cfg


def build_lstm_encoder(train, cfg, weights=None):
    print(train.shape)
    inp = Input(shape=(cfg['sequence_length'], 1))
    encoded = LSTM(32, return_sequences=True, dropout=0.3)(inp, training=True)
    # encoded = Dropout(0.3)(encoded)

    decoded = LSTM(32, return_sequences=True, dropout=0.3)(encoded, training=True)
    # decoded = Dropout(0.3)(decoded)
    out = TimeDistributed(Dense(1))(decoded)
    decoder = Model(inp, out, name='autoencoder')

    decoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    decoder.summary()

    decoder.fit(train, train, epochs=100, batch_size=64, verbose=2, shuffle=True)
    encoder = Model(inp, encoded, name='encoder')

    """
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    """
    return encoder, decoder, cfg


def build_cnn_encoder(train_x, cfg, weights=None):
    inp = Input(shape=(cfg['sequence_length'], 1))
    encoded = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(inp)
    encoded = Dropout(0.3)(encoded)
    encoded = MaxPooling1D(2, padding="same")(encoded)  # 5 dims
    encoded = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 5 dims
    encoded = MaxPooling1D(2, padding="same")(encoded)  # 3 dims
    encoded = Dropout(0.3)(encoded)
    cfg['decoded_features_shape'] = encoded.shape[1]

    decoded = Conv1D(1, 3, activation="relu", padding="same")(encoded)
    decoded = Dropout(0.3)(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Conv1D(filters=16, kernel_size=3, activation='relu', padding="same")(decoded)
    decoded = Dropout(0.3)(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = TimeDistributed(Dense(1))(decoded)
    decoder = Model(inp, decoded, name='autoencoder')
    decoder.summary()

    decoder.compile(optimizer=Lookahead(RAdam()), loss='mse')
    decoder.fit(train_x, train_x, epochs=5, batch_size=16, verbose=2, shuffle=True)
    encoder = Model(inp, encoded, name='encoder')

    return encoder, decoder, cfg


def build_rnn_encoder(train_x, cfg, weights=None):
    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
    x = LSTM(10, dropout=0.4)(inp)
    encoded = RepeatVector(cfg['sequence_length'])(x)
    x = LSTM(10, return_sequences=True, dropout=0.4)(encoded)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.4)(x)
    decoded = TimeDistributed(Dense(1, activation='relu'))(x)

    decoder = Model(inp, decoded, name='autoencoder')
    decoder.summary()

    decoder.compile(optimizer=Lookahead(RAdam()), loss='mse')
    decoder.fit(train_x, train_x, epochs=10, batch_size=64, verbose=2, shuffle=True)
    encoder = Model(inp, encoded, name='encoder')
    encoder.summary()
    return encoder, decoder, cfg
