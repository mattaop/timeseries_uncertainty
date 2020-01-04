import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras_lookahead import Lookahead


def build_model(train_x, train_y, cfg, val_x, val_y):
    print('cnn_model')
    filters = 64
    number_of_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    patience = 4000
    dropout_rate = 0.4

    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))

    x = Conv1D(filters, kernel_size=2, activation='relu')(inp)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters, kernel_size=2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters, kernel_size=2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters, kernel_size=2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1)(x)

    model = Model(inp, x, name=cfg['model'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=patience)
    cb = [es]
    model.compile(optimizer=Lookahead(Adam(lr=learning_rate)), loss='mse')
    model.summary()

    history = model.fit(train_x, train_y, epochs=number_of_epochs, batch_size=batch_size, callbacks=cb,
                        validation_data=(val_x, val_y),
                        verbose=0)

    """
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.title('CNN log loss')
    plt.ylabel('Log loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    """

    return model
