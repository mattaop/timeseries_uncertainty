from keras import Model
from keras.layers import *
from keras_radam import RAdam
from keras_lookahead import Lookahead
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class Autoencoder:
    def __init__(self, data, cfg):
        self.sequence_length = cfg['sequence_length']
        self.encoder_type = cfg['autoencoder'].lower()
        self.epochs = 60
        self.encoder_output_dim = None
        self.batch_size = 16
        train_x, _, _ = data.get_train_sequence()
        test_x, _, _ = data.get_test_sequence()
        self.train_x = train_x
        self.test_x = test_x
        self.encoder = None
        self.decoder = None
        self.build_autoencoder()

    def build_autoencoder(self):
        if self.encoder_type == 'lstm':
            self.build_lstm_encoder()
        elif self.encoder_type == 'cnn':
            self.build_cnn_encoder()
        elif self.encoder_type == 'new_lstm':
            self.build_new_lstm_encoder()

    def build_lstm_encoder(self):
        inp = Input(shape=(self.sequence_length, 1))
        encoded = LSTM(30, return_sequences=True, dropout=0.3)(inp, training=True)
        # encoded = Dropout(0.3)(encoded)
        self.encoder_output_dim = 30
        decoded = LSTM(32, return_sequences=True, dropout=0.3)(encoded, training=True)
        # decoded = Dropout(0.3)(decoded)
        out = TimeDistributed(Dense(1))(decoded)
        self.decoder = Model(inp, out, name='autoencoder')

        self.decoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
        self.decoder.summary()
        self.encoder = Model(inp, encoded, name='encoder')

    def build_new_lstm_encoder(self):
        inputs = Input(shape=(self.sequence_length, 1))
        encoded = LSTM(10)(inputs)
        self.encoder_output_dim = 10
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(1, return_sequences=True)(decoded)
    
        self.decoder = Model(inputs, decoded)
        self.decoder.compile(optimizer='adam', loss='mse')
        self.decoder.summary()
        self.encoder = Model(inputs, encoded)

    def build_cnn_encoder(self):
        inp = Input(shape=(self.sequence_length, 1))
        encoded = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(inp)
        encoded = Dropout(0.3)(encoded)
        encoded = MaxPooling1D(2, padding="same")(encoded)  # 5 dims
        encoded = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 5 dims
        encoded = MaxPooling1D(2, padding="same")(encoded)  # 3 dims
        encoded = Dropout(0.3)(encoded)

        decoded = Conv1D(1, 3, activation="relu", padding="same")(encoded)
        decoded = Dropout(0.3)(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(filters=16, kernel_size=3, activation='relu', padding="same")(decoded)
        decoded = Dropout(0.3)(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = TimeDistributed(Dense(1))(decoded)
        self.decoder = Model(inp, decoded, name='autoencoder')
        self.decoder.summary()

        self.decoder.compile(optimizer='adam', loss='mse')
        self.encoder = Model(inp, encoded, name='encoder')

    def train(self):
        # Train autoencoder
        self.decoder.fit(self.train_x, self.train_x, epochs=self.epochs, batch_size=self.batch_size, verbose=2,
                         shuffle=True)

    def test(self):
        # Test autoencoder on holdout data
        predictions = self.decoder.predict(self.test_x)

        mse = 0
        for i in range(len(predictions)):
            mse += mean_squared_error(self.test_x[i], predictions[i])
        print('Test mean mse:', mse/len(predictions))
