import numpy as np

class Avocado:
    def __init__(self, df, cfg):
        self.x_train_conv = None
        self.x_train_org = None
        self.x_test_conv = None
        self.x_test_org = None
        self.x_holdout_train_conv = None
        self.x_holdout_train_org = None
        self.x_holdout_test_conv = None
        self.x_holdout_test_org = None
        self.x_holdout_conv = None
        self.x_holdout_org = None
        self.x_train = None
        self.x_test = None
        self.X = None

        self.y_train_conv = None
        self.y_train_org = None
        self.y_test_conv = None
        self.y_test_org = None
        self.y_holdout_train_conv = None
        self.y_holdout_train_org = None
        self.y_holdout_test_conv = None
        self.y_holdout_test_org = None
        self.y_holdout_conv = None
        self.y_holdout_org = None
        self.y_train = None
        self.y_test = None
        self.Y = None

        self.f_train_conv = None
        self.f_train_org = None
        self.f_test_conv = None
        self.f_test_org = None
        self.f_holdout_train_conv = None
        self.f_holdout_train_org = None
        self.f_holdout_test_conv = None
        self.f_holdout_test_org = None
        self.f_holdout_conv = None
        self.f_holdout_org = None
        self.f_train = None
        self.f_test = None
        self.F = None

        self.data_split(df, self.gen_sequence, ['AveragePrice'], cfg, prefix='x')
        self.data_split(df, self.gen_labels, ['AveragePrice'], cfg, prefix='y')
        self.data_split(df, self.gen_sequence, ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags',
                                                'Large Bags', 'XLarge Bags'], cfg, prefix='f')
        print('X.shape: ', self.X.shape)
        print('Y.shape: ', self.Y.shape)
        print('F.shape: ', self.F.shape)

    def gen_sequence(self, id_df, seq_length, cols):
        data_matrix = id_df[cols].values
        num_elements = data_matrix.shape[0]

        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    def gen_labels(self, id_df, seq_length, label):

        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]

        return data_matrix[seq_length:num_elements, :]

    def data_split(self, df, sequence_generator, cols, cfg, prefix):
        train_conv, test_conv = [], []
        train_org, test_org = [], []
        holdout_train_conv, holdout_test_conv = [], []
        holdout_train_org, holdout_test_org = [], []
        for county in df["region"].unique():
            for sequence in sequence_generator(
                    df[np.logical_and.reduce(
                        [df["region"] == county, df["type"] == "conventional", df["year"] != 2018, ])],
                    cfg['sequence_length'], cols):
                train_conv.append(sequence) if county != 'Albany' else holdout_train_conv.append(sequence)
            for sequence in sequence_generator(
                    df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] != 2018, ])],
                    cfg['sequence_length'], cols):
                train_org.append(sequence) if county != 'Albany' else holdout_train_org.append(sequence)
            for sequence in sequence_generator(
                    df[np.logical_and.reduce(
                        [df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
                    cfg['sequence_length'], cols):
                test_conv.append(sequence) if county != 'Albany' else holdout_test_conv.append(sequence)
            for sequence in sequence_generator(
                    df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
                    cfg['sequence_length'], cols):
                test_org.append(sequence) if county != 'Albany' else holdout_test_org.append(sequence)
        if prefix == 'x':
            self.x_train_conv, self.x_train_org = np.asarray(train_conv), np.asarray(train_org)
            self.x_test_conv, self.x_test_org = np.asarray(test_conv), np.asarray(test_org)
            self.x_holdout_train_conv, self.x_holdout_test_conv = np.asarray(holdout_train_conv), np.asarray(holdout_test_conv)
            self.x_holdout_train_org, self.x_holdout_test_org = np.asarray(holdout_train_org), np.asarray(holdout_test_org)
            self.x_train = np.concatenate([self.x_train_conv, self.x_train_org], axis=0)
            self.x_test = np.concatenate([self.x_test_conv, self.x_test_org], axis=0)
            self.x_holdout_conv = np.concatenate([self.x_holdout_train_conv, self.x_holdout_test_conv], axis=0)
            self.x_holdout_org = np.concatenate([self.x_holdout_train_org, self.x_holdout_test_org], axis=0)
            self.X = np.concatenate([self.x_train, self.x_test], axis=0)
        elif prefix == 'y':
            self.y_train_conv, self.y_train_org = np.asarray(train_conv), np.asarray(train_org)
            self.y_test_conv, self.y_test_org = np.asarray(test_conv), np.asarray(test_org)
            self.y_holdout_train_conv, self.y_holdout_test_conv = np.asarray(holdout_train_conv), np.asarray(holdout_test_conv)
            self.y_holdout_train_org, self.y_holdout_test_org = np.asarray(holdout_train_org), np.asarray(holdout_test_org)
            self.y_train = np.concatenate([self.y_train_conv, self.y_train_org], axis=0)
            self.y_test = np.concatenate([self.y_test_conv, self.y_test_org], axis=0)
            self.y_holdout_conv = np.concatenate([self.y_holdout_train_conv, self.y_holdout_test_conv], axis=0)
            self.y_holdout_org = np.concatenate([self.y_holdout_train_org, self.y_holdout_test_org], axis=0)
            self.Y = np.concatenate([self.y_train, self.y_test ], axis=0)
        elif prefix == 'f':
            self.f_train_conv, self.f_train_org = np.asarray(train_conv), np.asarray(train_org)
            self.f_test_conv, self.f_test_org = np.asarray(test_conv), np.asarray(test_org)
            self.f_holdout_train_conv, self.f_holdout_test_conv = np.asarray(holdout_train_conv), np.asarray(holdout_test_conv)
            self.f_holdout_train_org, self.f_holdout_test_org = np.asarray(holdout_train_org), np.asarray(holdout_test_org)
            self.f_train = np.concatenate([self.f_train_conv, self.f_train_org], axis=0)
            self.f_test = np.concatenate([self.f_test_conv, self.f_test_org], axis=0)
            self.f_holdout_conv = np.concatenate([self.f_holdout_train_conv, self.f_holdout_test_conv], axis=0)
            self.f_holdout_org = np.concatenate([self.f_holdout_train_org, self.f_holdout_test_org], axis=0)
            self.F = np.concatenate([self.f_train, self.f_test], axis=0)
