from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping


def build_model(train_x, train_y, cfg):
    inp = Input(shape=(cfg['sequence_length'], cfg['n_feature_extraction']+1))
    # x = Dropout(cfg['dropout_rate'])(inp)
    x = LSTM(128, activation='relu', return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    # x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(x)
    # x = Dropout(cfg['dropout_rate'])(x)
    # x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(x)
    # x = Dropout(cfg['dropout_rate'])(x)
    x = LSTM(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(cfg['forecasting_horizon'])(x)

    model = Model(inp, x, name=cfg['model'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=cfg['patience'])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'], callbacks=[es],
              validation_split=0.1, verbose=2)
    return model


