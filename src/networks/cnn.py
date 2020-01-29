from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras_lookahead import Lookahead


def build_model(train_x, train_y, cfg, val_x, val_y):
    filters = cfg['number_of_nodes']
    number_of_epochs = cfg['number_of_epochs']
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    dropout_rate = cfg['dropout_rate']
    number_of_layers = cfg['number_of_layers']

    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))

    x = Conv1D(filters, kernel_size=2, activation='relu')(inp)
    x = Dropout(dropout_rate)(x)

    if number_of_layers > 1:
        x = Conv1D(filters, kernel_size=2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(pool_size=2)(x)

    if number_of_layers > 2:
        x = Conv1D(filters, kernel_size=2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1)(x)

    model = Model(inp, x, name=cfg['model'])
    model.compile(optimizer=Lookahead(Adam(lr=learning_rate)), loss='mse')
    model.summary()

    model.fit(train_x, train_y, epochs=number_of_epochs, batch_size=batch_size, validation_data=(val_x, val_y),
              verbose=2)

    return model
