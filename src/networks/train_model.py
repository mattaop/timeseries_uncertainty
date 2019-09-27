from src.networks import autoencoder, CNN, LSTM, ResNet


def train_model(train_x, train_y, cfg):
    if cfg['model'].lower() == 'resnet':
        model = ResNet.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'cnn':
        model = CNN.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'lstm':
        model = LSTM.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'autoencoder':
        model = autoencoder.build_model(train_x, train_y, cfg)
    else:
        ModuleNotFoundError('Model', cfg['model'], 'does not exist')
        model = None
    return model
