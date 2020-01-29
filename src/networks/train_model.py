from src.networks import cnn, lstm, rnn


def train_model(train_x, train_y, cfg, val_x=None, val_y=None):
    if cfg['model'].lower() == 'cnn':
        model = cnn.build_model(train_x, train_y, cfg['cnn_hyperparameters'], val_x, val_y)
    elif cfg['model'].lower() == 'lstm':
        model = lstm.build_model(train_x, train_y, cfg['lstm_hyperparameters'], val_x, val_y)
    elif cfg['model'].lower() == 'rnn':
        model = rnn.build_model(train_x, train_y, cfg['rnn_hyperparameters'], val_x, val_y)
    else:
        ModuleNotFoundError('Model', cfg['model'], 'does not exist')
        model = None
    return model
