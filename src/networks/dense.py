from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_radam import RAdam
from keras_lookahead import Lookahead


def build_model(train_x, train_y, cfg):
    print(train_x.shape)
    print(train_y.shape)
    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
    x = Dense(cfg['number_of_nodes'], activation='relu')(inp)
    x = Dropout(0.3)(x, training=True)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = Dropout(0.3)(x, training=True)
    out = Dense(cfg['forecasting_horizon'])(x)

    model = Model(inp, out, name=cfg['model'])
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
    checkpoint = ModelCheckpoint('weights\\LSTM_weights_{epoch:02d}_{val_loss:.2f}.hdf5', monitor='val_loss',
                                 verbose=0, save_best_only=True, mode='min')

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'],
              shuffle=True,
              callbacks=[early_stopping, checkpoint],
              validation_split=0.1,
              verbose=2)
    return model

