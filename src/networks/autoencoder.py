from keras import Input, Model
from keras.layers import Dense, LSTM, RepeatVector
from src.utility.permanent_dropout import permanent_dropout


def build_model(train_x, train_y, cfg):
    inp = Input(shape=(cfg['sequence_length'], 1))
    x = LSTM(cfg['number_of_nodes'], activation='relu')(inp)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = RepeatVector(cfg['sequence_length'])(x)
    x = LSTM(cfg['number_of_nodes'], activation='relu', return_sequences=False)(x)
    x = permanent_dropout(cfg['dropout_rate'])(x)
    x = Dense(1)(x)
    model = Model(inp, x, name=cfg['model'])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, epochs=cfg['number_of_epochs'], batch_size=cfg['batch_size'], callbacks=[],
              validation_split=0.1, verbose=2)
    return model
