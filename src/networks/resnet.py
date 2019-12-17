from keras import Input, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from src.utility.unused_files.residual_layer import residual_layer
from src.utility.unused_files.permanent_dropout import permanent_dropout


def build_model(train_x, train_y, cfg):
    inp = Input(shape=(cfg['sequence_length'], 1))
    x = residual_layer(inp, Conv1D(cfg['number_of_nodes'], kernel_size=2, activation='relu', padding='same'))
    # x = permanent_dropout(cfg['dropout_rate'])(x)
    x = residual_layer(x, Conv1D(cfg['number_of_nodes'], kernel_size=2, activation='relu', padding='same'))
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = residual_layer(x, Conv1D(cfg['number_of_nodes'], kernel_size=2, activation='relu', padding='same'))
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = residual_layer(x, Conv1D(cfg['number_of_nodes'], kernel_size=2, activation='relu', padding='same'))
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(cfg['number_of_nodes'], activation='relu')(x)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = Dense(1)(x)

    model = Model(inp, x, name=cfg['model'])
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'], callbacks=[],
              validation_split=0.1, verbose=2)
    return model
