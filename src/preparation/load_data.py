import pandas as pd
from src.preparation.config.open_config import load_config_file


def load_raw_data(file_name='M_train//Daily-train.csv'):
    return pd.read_csv('C://Users//mathi//PycharmProjects//timeseries_uncertainty//data//raw//' + file_name, header=0, index_col=0)


def load_processed_data(file_name='M_train//Daily-train.csv'):
    return pd.read_csv('data//processed//' + file_name, header=0, index_col=0)


def load_data(data_set=None):
    if not data_set:
        cfg = load_config_file(
            'C://Users//mathi//PycharmProjects//timeseries_uncertainty//src//preparation//config\\config.yml',
            print_config=True)
        data_set = cfg['data_source'].lower()

    if data_set.lower() == 'airpassengers':
        cfg = load_config_file(
            'C://Users//mathi//PycharmProjects//timeseries_uncertainty//src//preparation//config\\airpassengers.yml',
            print_config=True)
        df = load_raw_data(file_name='AirPassengers.csv')
        df.rename(columns={"#Passengers": "y"}, inplace=True)

    elif data_set.lower() == 'avocado':
        cfg = load_config_file(
            'C://Users//mathi//PycharmProjects//timeseries_uncertainty//src//preparation//config\\avocado.yml',
            print_config=True)
        df = load_raw_data(file_name='avocado.csv')
        print(df)

    elif data_set.lower() == 'oslo_temperature' or data_set.lower() == 'oslo' or data_set.lower() == 'temperature':
        cfg = load_config_file(
            'C://Users//mathi//PycharmProjects//timeseries_uncertainty//src//preparation//config\\oslo.yml',
            print_config=True)
        df = pd.read_csv('C://Users//mathi//PycharmProjects//timeseries_uncertainty//data//raw//OsloTemperature.csv',
                         header=0, sep=';', index_col=0)
        df.set_index('time', inplace=True)
        df.drop(columns=['station', 'id', 'max(air_temperature P1M)', 'min(air_temperature P1M)'], inplace=True)
        df.dropna(how='any', inplace=True)
        df.rename(columns={"mean(air_temperature P1M)": "y"}, inplace=True)
        idx = pd.date_range('1937-01-31', freq='M', periods=994)
        for index, row in df.iterrows():
            df.loc[index, 'y'] = df.loc[index, 'y'].replace(',', '.')
        # df = pd.DataFrame(data=df['mean_temperature'].index[1:], index=idx, columns=['mean_temperature'])
        df.set_index(idx, inplace=True)
        df.drop(df.index[0], inplace=True)
        df["y"] = pd.to_numeric(df["y"])

    else:
        cfg = load_config_file(
            'C://Users//mathi//PycharmProjects//timeseries_uncertainty//src//preparation//config\\config.yml',
            print_config=True)
        ImportError('Model', cfg['model'], 'is not a data source.')
        df = None
    return df, cfg


def main():
    df, cfg = load_data(data_set='avocado')
    print(df)


if __name__ == '__main__':
    main()
