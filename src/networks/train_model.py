from src.networks import autoencoder, cnn, lstm, resnet, cnn_lstm


def train_model(train_x, train_y, cfg):
    if cfg['model'].lower() == 'resnet':
        model = resnet.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'cnn':
        model = cnn.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'lstm':
        model = lstm.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'autoencoder':
        model = autoencoder.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'cnn_lstm_encoder':
        model = cnn_lstm.build_model(train_x, train_y, cfg)
    else:
        ModuleNotFoundError('Model', cfg['model'], 'does not exist')
        model = None
    return model
