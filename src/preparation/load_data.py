import pandas as pd


def load_raw_data(file_name='M_train//Daily-train.csv'):
    return pd.read_csv('C://Users//mathi//PycharmProjects//timeseries_uncertainty//data//raw//' + file_name, header=0, index_col=0)


def load_processed_data(file_name='M_train//Daily-train.csv'):
    return pd.read_csv('data//processed//' + file_name, header=0, index_col=0)
