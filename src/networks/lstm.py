from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_radam import RAdam
from keras_lookahead import Lookahead


def build_model(train_x, train_y, cfg):
    print(train_x.shape)
    print(train_y.shape)
    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
    # x = Dropout(cfg['dropout_rate'])(inp)
    x = LSTM(128, return_sequences=True, dropout=0.3)(inp, training=True)
    # x = Dropout(0.3)(x)
    # x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(x)
    # x = Dropout(cfg['dropout_rate'])(x)
    # x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(x)
    # x = Dropout(cfg['dropout_rate'])(x)
    x = LSTM(32, return_sequences=False, dropout=0.3)(x, training=True)
    # x = Dropout(0.3)(x)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    # x = Dropout(0.3)(x)
    out = Dense(cfg['forecasting_horizon'])(x)

    model = Model(inp, out, name=cfg['model'])
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=cfg['patience'])
    #checkpoint = ModelCheckpoint('weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
    #                             verbose=0, save_best_only=True, mode='max')

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'],
              shuffle=True,
              # validation_split=0.1,
              verbose=2)
    return model


