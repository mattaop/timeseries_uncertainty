from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_radam import RAdam
from keras.optimizers import Adam
from keras.regularizers import l2
from keras_lookahead import Lookahead
import matplotlib.pyplot as plt
import numpy as np


def build_model(train_x, train_y, cfg, val_x, val_y):
    nodes = 64
    number_of_epochs = 300
    batch_size = 64
    learning_rate = 0.001
    patience = 3000
    dropout_rate = 0.4
    number_of_lstm_layers = 3

    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
    x = inp
    for i in range(number_of_lstm_layers-1):
        x = LSTM(nodes, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
    x = LSTM(nodes, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1)(x)

    model = Model(inp, out, name=cfg['model'])
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience)
    #checkpoint = ModelCheckpoint('weights\\LSTM_weights_{epoch:02d}_{val_loss:.2f}.hdf5', monitor='val_loss',
    #                             verbose=0, save_best_only=True, mode='min')

    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    model.summary()
    history = model.fit(train_x, train_y, epochs=number_of_epochs, batch_size=batch_size,
              shuffle=True,
              callbacks=[early_stopping],
              # validation_split=0.1,
              validation_data=(val_x, val_y),
              verbose=2)

    return model


