import numpy as np
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint


class MonteCarloNetwork:
    def __init__(self, data, autoencoder, cfg):
        self.sequence_length = cfg['sequence_length']
        self.network_type = cfg['model'].lower()
        self.epochs = cfg['number_of_epochs']
        self.batch_size = cfg['batch_size']
        self.num_forward_passes = cfg['number_of_mc_forward_passes']
        self.num_features = data.num_features
        self.patience = cfg['patience']
        self.forecasting_horizon = cfg['forecasting_horizon']
        self.dropout_test = 0.5
        train_x, train_f, train_y = data.get_train_sequence()
        self.autoencoder = autoencoder
        self.external_features = cfg['external_features']
        self.scaler = StandardScaler()
        if self.external_features:
            if self.network_type is 'lstm':
                self.model = self.build_model_lstm(np.concatenate([train_x, train_f]))
            else:
                self.model = self.build_model_cnn(np.concatenate([train_x, train_f]))
        else:
            if self.network_type is 'lstm':
                self.model = self.build_model_lstm(train_x)
            else:
                self.model = self.build_model_cnn(train_x)

    def build_model_lstm(self, train_x):
        inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
        # x = Dropout(cfg['dropout_rate'])(inp)
        x = LSTM(32, return_sequences=True, dropout=0.3)(inp, training=True)
        x = Dropout(0.3)(x)
        # x = LSTM(32, return_sequences=True)(x)
        # x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False, dropout=0.3)(x, training=True)
        x = Dropout(0.3)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.3)(x, training=True)
        out = Dense(self.forecasting_horizon)(x)

        model = Model(inp, out, name='LSTM')

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def build_model_cnn(self, train_x):
        inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
        x = Conv1D(64, kernel_size=2, activation='relu')(inp)
        x = Dropout(0.3)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, kernel_size=2, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.forecasting_horizon)(x)

        model = Model(inp, x, name='CNN')

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def train(self, x, y, f=None):
        if self.external_features:
            if self.autoencoder:
                enc_pred = self.autoencoder.encoder.predict(x)
                x = np.concatenate([enc_pred, f], axis=2)
                x = self.scaler.fit_transform(x.reshape(-1, self.autoencoder.encoder_output_dim + self.num_features - 1)) \
                    .reshape(-1, self.sequence_length, self.autoencoder.encoder_output_dim + self.num_features - 1)
            else:
                x = np.concatenate([x, f], axis=2)
                x = self.scaler.fit_transform(x.reshape(-1, self.num_features)) \
                    .reshape(-1, self.num_features, self.num_features)
        else:
            if self.autoencoder:
                x = self.autoencoder.encoder.predict(x)
                x = self.scaler.fit_transform(x.reshape(-1, self.autoencoder.encoder_output_dim)) \
                    .reshape(-1, self.sequence_length, self.autoencoder.encoder_output_dim)
            else:
                x = self.scaler.fit_transform(x.reshape(-1, 1)) \
                    .reshape(-1, self.sequence_length, 1)

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=self.patience)
        checkpoint = ModelCheckpoint('weights\\LSTM_weights_{epoch:02d}_{val_loss:.2f}.hdf5', monitor='val_loss',
                                     verbose=0, save_best_only=True, mode='min')
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                       callbacks=[early_stopping, checkpoint], validation_split=0.1, verbose=2)

    def stochastic_dropout(self, x, f=None):
        if self.autoencoder:
            enc = K.function([self.autoencoder.encoder.layers[0].input, K.learning_phase()],
                             [self.autoencoder.encoder.layers[-1].output])
            enc_pred = np.vstack(enc([x, self.dropout_test]))
            if self.external_features:
                enc_pred = np.concatenate([enc_pred, f], axis=2)
                trans_input = self.scaler.transform(
                    enc_pred.reshape(-1, self.autoencoder.encoder_output_dim + self.num_features - 1)) \
                    .reshape(-1, self.sequence_length, self.autoencoder.encoder_output_dim + self.num_features - 1)
            else:
                trans_input = self.scaler.transform(enc_pred.reshape(-1, self.autoencoder.encoder_output_dim)) \
                    .reshape(-1, self.sequence_length, self.autoencoder.encoder_output_dim)
        else:
            if self.external_features:
                x_input = np.concatenate([x, f], axis=2)
                trans_input = self.scaler.transform(x_input.reshape(-1, self.num_features))\
                    .reshape(-1, self.sequence_length, self.num_features)
            else:
                trans_input = self.scaler.transform(x.reshape(-1, 1)).reshape(-1, self.sequence_length, 1)

        NN = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])

        NN_pred = NN([trans_input, self.dropout_test])

        return np.vstack(NN_pred)
