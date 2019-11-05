from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping


def build_model(train_x, train_y, cfg):
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    inp = Input(shape=(cfg['sequence_length'], 1))
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = RepeatVector(cfg['forecasting_horizon'])(x)
    x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(cfg['number_of_nodes']))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Dense(1))(x)
    model = Model(inp, x, name=cfg['model'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=cfg['patience'])
    cb = [es]
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'], callbacks=cb,
              validation_split=0.1, verbose=2)
    return model
