import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preparation.load_data import load_data


class Avocado:
    def __init__(self, cfg):
        self.path = cfg['data_path']+'avocado.csv'
        self.seq_length = cfg['sequence_length']
        self.features = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags',
                         'XLarge Bags']
        self.target_feature = ['AveragePrice']
        self.test_split = cfg['test_size']
        self.data = self._load_data()
        self.num_features = 9
        self.regions = self.data['region'].unique()
        self.avocado_types = self.data['type'].unique()
        self.holdout_region = ['Albany']
        self._init_data_processing(self.data, cfg)
        # self._transformations()
        self.train = self.data[:int(len(self.data) * (1 - self.test_split))]
        self.test = self.data[int(len(self.data) * (1 - self.test_split) - self.seq_length):]

    def _load_data(self):
        return pd.read_csv(self.path, header=0, index_col=0)

    def _init_data_processing(self, data, cfg):
        data['Date'] = pd.to_datetime(data['Date'])
        cfg['target_feature'] = 'AveragePrice'
        # df = df.pivot_table(values='AveragePrice', index='Date', columns=['region', 'type'], aggfunc='mean')
        data = data.pivot_table(index='Date', columns=['region', 'type'], aggfunc='mean')
        data = data.fillna(method='backfill').dropna()
        # albany_conventional = extract_external_features(df, region='Albany', avocado_type='conventional')
        cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags',
                'XLarge Bags']
        data = data[cols]
        cfg['num_features'] = len(cols)
        self.data = data

    def _transformations(self):
        # Log data to remove change in variance
        log_data = np.log(self.data)
        log_data = log_data.replace(-np.inf, 0)
        # Differentiate data to remove change in mean
        log_diff_data = log_data.diff(periods=1, axis=0)
        self.data = self.data.drop(self.data.index[0])
        self.data[self.data.columns.values] = log_diff_data.drop(log_diff_data.index[0]).values

    def _gen_sequence(self, data, feature, region='Albany', avocado_type='conventional'):
        df = data.loc[:, (feature, region, avocado_type)]
        data_matrix = df.values
        num_elements = data_matrix.shape[0]
        time_series_data = np.zeros([num_elements-self.seq_length, self.seq_length, len(feature)])
        for i in range(num_elements-self.seq_length):
            time_series_data[i, :, :] = data_matrix[i:i+self.seq_length]
        return time_series_data

    def _gen_targets(self, data, feature, region='Albany', avocado_type='conventional'):
        df = data.loc[:, (feature, region, avocado_type)]
        data_matrix = df.values
        targets = data_matrix[self.seq_length:]
        return targets.reshape(targets.shape[0], 1)

    def _get_sequence_data(self, data, regions, avocado_types):
        x, f, y = [], [], []
        for region in regions:
            for avocado_type in avocado_types:
                x.append(self._gen_sequence(data, self.target_feature, region, avocado_type))
                f.append(self._gen_sequence(data, self.features, region, avocado_type))
                y.append(self._gen_targets(data, self.target_feature, region, avocado_type))
        x = np.asarray(x)
        print(x.shape)
        x = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])
        f = np.asarray(f)
        f = f.reshape([f.shape[0]*f.shape[1], f.shape[2], f.shape[3]])
        y = np.asarray(y)
        y = y.reshape([y.shape[0] * y.shape[1], y.shape[2]])
        return x, f, y

    def get_train_sequence(self):
        regions = self.regions[self.regions != self.holdout_region]
        return self._get_sequence_data(self.train, regions, self.avocado_types)

    def get_test_sequence(self):
        regions = self.regions[self.regions != self.holdout_region]
        return self._get_sequence_data(self.test, regions, self.avocado_types)

    def get_holdout_sequence(self, avocado_types):
        return self._get_sequence_data(self.data, self.holdout_region, avocado_types)

    def get_x(self):
        x = []
        for region in self.regions:
            if region is not self.holdout_region:
                region_conv = self._gen_sequence(self.data, self.target_feature, region, 'conventional')
                region_org = self._gen_sequence(self.data, self.target_feature, region, 'organic')
                x.append(region_conv)
                x.append(region_org)
        x = np.asarray(x)
        x = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])
        train_x = x[:int(len(x)*(1-self.test_split))]
        test_x = x[int(len(x)*(1-self.test_split)):]
        return train_x, test_x

    def get_features(self):
        f = []
        for region in self.regions:
            if region is not self.holdout_region:
                region_conv = self._gen_sequence(self.data, self.features, region, 'conventional')
                region_org = self._gen_sequence(self.data, self.features, region, 'organic')
                f.append(region_conv)
                f.append(region_org)
        f = np.asarray(f)
        f = f.reshape([f.shape[0]*f.shape[1], f.shape[2], len(self.features)])
        train_f = f[:int(len(f)*(1-self.test_split))]
        test_f = f[int(len(f)*(1-self.test_split)):]
        return train_f, test_f

    def get_y(self):
        y = []
        for region in self.regions:
            if region is not self.holdout_region:
                region_conv = self._gen_targets(self.data, self.target_feature, region, 'conventional')
                region_org = self._gen_targets(self.data, self.target_feature, region, 'organic')
                y.append(region_conv)
                y.append(region_org)
        y = np.asarray(y)
        y = y.reshape([y.shape[0] * y.shape[1], y.shape[2]])
        train_y = y[:int(len(y) * (1 - self.test_split))]
        test_y = y[int(len(y) * (1 - self.test_split)):]
        return train_y, test_y

    def get_holdout(self):
        x, y, f = [], [], []
        for region in self.holdout_region:
            # x_conv = self._gen_sequence(self.target_feature, region, 'conventional')
            x_org = self._gen_sequence(self.data, self.target_feature, region, 'organic')
            # x.append(x_conv)
            x.append(x_org)

            # y_conv = self._gen_targets(self.target_feature, region, 'conventional')
            y_org = self._gen_targets(self.data, self.target_feature, region, 'organic')
            # y.append(y_conv)
            y.append(y_org)

            # f_conv = self._gen_sequence(self.features, region, 'conventional')
            f_org = self._gen_sequence(self.data, self.features, region, 'organic')
            # f.append(f_conv)
            f.append(f_org)
        x = np.asarray(x)
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3]])
        y = np.asarray(y)
        y = y.reshape([y.shape[0] * y.shape[1], y.shape[2]])
        f = np.asarray(f)
        f = f.reshape([f.shape[0] * f.shape[1], f.shape[2], len(self.features)])

        return x, f, y

    # def get_series(self, feature):

    def plot_series(self, region, avocado_type, feature='AveragePrice'):
        plt.figure()
        plt.plot(self.data.loc[:, (feature, region, avocado_type)])
        plt.show()


def main():
    df, cfg = load_data()
    avocado = Avocado(cfg)
    print(len(avocado.data))
    avocado.plot_series('Albany', 'organic')
    train_x, _ = avocado.get_x()
    train_y, _ = avocado.get_y()
    train_f, _ = avocado.get_features()
    x, f, y = avocado.get_train_sequence()
    print(train_x.shape, x.shape)
    # print(avocado.get_test_sequence())
    x, f, y = avocado.get_holdout_sequence(['organic'])
    x2, f2, y2 = avocado.get_holdout()
    print(np.array_equal(x, x2))
    print(np.array_equal(f, f2))
    print(np.array_equal(y, y2))
    print(avocado.train['AveragePrice'].shape)


if __name__ == '__main__':
    main()

