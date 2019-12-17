import numpy as np
from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_radam import RAdam
from keras.optimizers import Adam
from keras.regularizers import l2, l1_l2
from keras_lookahead import Lookahead
import matplotlib.pyplot as plt


def build_model(train_x, train_y, cfg, val_x, val_y):
    nodes = 64
    number_of_epochs = 1500
    batch_size = 64
    learning_rate = 0.001
    patience = 1200
    dropout_rate = 0.4
    number_of_recurrent_layers = 1

    inp = Input(shape=(train_x.shape[1], train_x.shape[2]))
    x = inp
    for i in range(number_of_recurrent_layers-1):
        x = SimpleRNN(nodes, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
    x = SimpleRNN(nodes, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1)(x)

    model = Model(inp, out, name=cfg['model'])
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience)
    # checkpoint = ModelCheckpoint('weights\\RNN_weights_best.hdf5', monitor='val_loss',
    #                             verbose=0, save_best_only=True, mode='min')

    model.compile(optimizer=Lookahead(Adam(lr=learning_rate)), loss='mse')
    model.summary()
    history = model.fit(train_x, train_y, epochs=number_of_epochs, batch_size=batch_size,
                        shuffle=True,
                        callbacks=[early_stopping],
                        # validation_split=0.15,
                        validation_data=[val_x, val_y],
                        verbose=2)

    # summarize history for loss
    """
    plt.figure()
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.title('Model log loss')
    plt.ylabel('Log loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    """

    return model


