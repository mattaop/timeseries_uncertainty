from keras import Input, Model
from keras.layers import Dense, LSTM
from src.utility.permanent_dropout import permanent_dropout


def build_model(train_x, train_y, cfg):
    # unpack config
    # define model

    inp = Input(shape=(cfg['sequence_length'], 1))
    x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(inp)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=True)(x)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = LSTM(cfg['number_of_nodes'], activation='relu')(x)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = Dense(1)(x)

    model = Model(inp, x, name=cfg['model'])
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'], callbacks=[],
              validation_split=0.1, verbose=2)
    return model


