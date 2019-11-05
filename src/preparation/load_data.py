import pandas as pd
from pydataset import data

from src.preparation.generate_data import generate_sine_data, generate_arp_data, generate_time_series_data
from src.modeling.config.open_config import load_config_file


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
    elif cfg['data_source'].lower() == 'trend_seasonal':
        df = generate_time_series_data()
        df.dropna(axis=1, how='all', inplace=True)
        # df = df[['y']].values
    elif cfg['data_source'].lower() == 'avocado':
        df = load_raw_data(file_name='avocado.csv')

    else:
        ImportError('Model', cfg['model'], 'is not a data source.')
        df = None
    return df


def main():
    cfg = load_config_file('C://Users//mathi//PycharmProjects//timeseries_uncertainty//src//modeling//config\\config.yml', print_config=True)
    df = load_data(cfg)
    print(df.sort_values(by='Date'))
    print(df.region.unique())
    df_albany = df.loc[df['region'] == 'Albany']
    print(df_albany.sort_values(by='Date'))


if __name__ == '__main__':
    main()
