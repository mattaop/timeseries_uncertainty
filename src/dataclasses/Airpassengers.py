import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydataset
from sklearn.preprocessing import StandardScaler

from src.preparation.load_data import load_data


class Airpassengers:
    def __init__(self, cfg):
        self.path = 'AirPassengers'
        self.seq_length = cfg['sequence_length']
        self.target_feature = ['AirPassengers']
        self.test_split = cfg['test_size']
        self.data = self._load_data()
        self.num_features = 1
        self.scaler = StandardScaler()
        self._init_data_processing(self.data)
        if cfg['differencing']:
            self._transformations()
        self.train = self.data[:int(len(self.data) * (1 - self.test_split))]
        self.test = self.data[int(len(self.data) * (1 - self.test_split) - self.seq_length):]

    def _load_data(self):
        return pydataset.data(self.path)

    def _init_data_processing(self, data):
        data['AirPassengers'] = self.scaler.fit_transform(data['AirPassengers'].values.reshape(-1, 1))
        data = data['AirPassengers']
        print(data)
        self.data = data

    def _transformations(self):
        self.data = np.log(self.data)
        self.data = self.data.diff(periods=1)
        self.data = self.data.drop(self.data.index[0])

    def _gen_sequence(self, data):
        df = data
        data_matrix = df.values.reshape(df.values.shape[0], 1)
        num_elements = data_matrix.shape[0]
        time_series_data = np.zeros([num_elements-self.seq_length, self.seq_length, 1])
        for i in range(num_elements-self.seq_length):
            time_series_data[i, :] = data_matrix[i:i+self.seq_length]
        return time_series_data

    def _gen_targets(self, data):
        df = data
        data_matrix = df.values
        targets = data_matrix[self.seq_length:]
        return targets.reshape(targets.shape[0], 1)

    def _get_sequence_data(self, data):
        x = self._gen_sequence(data)
        y = self._gen_targets(data)

        x = np.asarray(x)
        y = np.asarray(y)
        print(x.shape)
        print(y.shape)
        x = x.reshape([x.shape[0],  x.shape[1], x.shape[2]])
        y = y.reshape([y.shape[0],  y.shape[1]])
        return x, y, None

    def get_train_sequence(self):
        return self._get_sequence_data(self.train)

    def get_test_sequence(self):
        return self._get_sequence_data(self.test)

    def get_holdout_sequence(self, **kwargs):
        return self._get_sequence_data(self.test)

    def plot_series(self):
        plt.figure()
        plt.plot(self.data)
        plt.show()


def main():
    df, cfg = load_data()
    print(cfg)
    passengers = Airpassengers(cfg)
    print(len(passengers.data))
    print(len(passengers.train))
    print(len(passengers.test))
    passengers.plot_series()


if __name__ == '__main__':
    main()
