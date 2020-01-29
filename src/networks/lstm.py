from keras import Model
from keras.layers import *
from keras.optimizers import Adam


def build_model(train_x, train_y, cfg, val_x, val_y):
    nodes = cfg['number_of_nodes']
    number_of_epochs = cfg['number_of_epochs']
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    dropout_rate = cfg['dropout_rate']
    number_of_lstm_layers = cfg['number_of_layers']

    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
    x = inp
    for i in range(number_of_lstm_layers-1):
        x = LSTM(nodes, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
    x = LSTM(nodes, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nodes, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1)(x)

    model = Model(inp, out, name=cfg['model'])

    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    model.summary()
    model.fit(train_x, train_y, epochs=number_of_epochs, batch_size=batch_size,
              shuffle=True, validation_data=(val_x, val_y), verbose=2)

    return model


