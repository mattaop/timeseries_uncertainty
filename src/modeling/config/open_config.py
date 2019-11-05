import yaml


def load_config_file(file, print_config=False):
    with open(file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    if not cfg['autoencoder']:
        cfg['n_feature_extraction'] = 0
    print(cfg)
    cfg['num_features'] = 1
    return cfg
