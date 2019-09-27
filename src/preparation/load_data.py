import pandas as pd
from src.preparation.generate_data import generate_sine_data, generate_arp_data
from pydataset import data


def load_raw_data(file_name='M_train//Daily-train.csv'):
    return pd.read_csv('C://Users//mathi//PycharmProjects//timeseries_uncertainty//data//raw//' + file_name, header=0, index_col=0)


def load_processed_data(file_name='M_train//Daily-train.csv'):
    return pd.read_csv('data//processed//' + file_name, header=0, index_col=0)


def load_data(cfg):
    if cfg['data_source'].lower() == 'm4':
        df = load_raw_data()
        df.dropna(axis=1, how='all', inplace=True)
        # df = df[['V3']].values
    elif cfg['data_source'].lower() == 'airpassengers':
        df = data('AirPassengers')
        df.dropna(axis=1, how='all', inplace=True)
        df = df[['AirPassengers']].values
    elif cfg['data_source'].lower() == 'sine_data':
        df = generate_sine_data()
        df.dropna(axis=1, how='all', inplace=True)
        df = df[['y']].values
    elif cfg['data_source'].lower() == 'arp':
        df = generate_arp_data(p=5)
        df.dropna(axis=1, how='all', inplace=True)
        df = df[['y']].values
    else:
        ImportError('Model', cfg['model'], 'is not a data source.')
        df = None
    return df
