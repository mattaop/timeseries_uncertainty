import numpy as np
from keras import Model
from keras.layers import *
from keras.callbacks import EarlyStopping


def build_model(train_x, train_y, cfg):
    print('cnn_model')
    #inp = Input(shape=(cfg['sequence_length'], cfg['n_feature_extraction']+cfg['num_features']))
    if cfg['autoencoder']:
        inp = Input(shape=(6+cfg['num_features'], 1))
        inp_small = Input(shape=((6+cfg['num_features'])/2, 1))

    else:
        inp = Input(shape=(cfg['sequence_length'], cfg['num_features']))
        inp_small = Input(shape=(cfg['sequence_length'], cfg['num_features']))

    def cnn_layers(inp):
        x = Conv1D(cfg['number_of_nodes'], kernel_size=2, activation='relu')(inp)
        x = Dropout(0.4)(x)
        x = GlobalMaxPool1D(pool_size=2)(x)
        return x

    #x = Conv1D(cfg['number_of_nodes'], kernel_size=2, activation='relu')(x)
    #x = Dropout(0.4)(x)
    #x = MaxPooling1D(pool_size=2)(x)

    original = cnn_layers(inp)
    downsample_small = cnn_layers(inp_small)
    x = Flatten()(np.concatenate([original, downsample_small]))
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(cfg['forecasting_horizon'])(x)

    model = Model(inp, x, name=cfg['model'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=cfg['patience'])
    cb = [es]
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    downsample_small_x = train_x
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'], callbacks=cb,
              # validation_data=(val_x, val_y),
              # validation_split=0.15,
              verbose=2)
    return model
