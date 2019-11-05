import numpy as np


# Split data into training and testing
def train_test_split(df, test_split=0.2):
    test_size = int(test_split*len(df))
    return df[:-test_size], df[-test_size:]


# split a univariate sequence into samples
def split_sequence(sequence, cfg, num_features=1):
    x = np.zeros([len(sequence)-cfg['sequence_length']-cfg['forecasting_horizon']+1, cfg['sequence_length'], 1])
    y = np.zeros([len(sequence)-cfg['sequence_length']-cfg['forecasting_horizon']+1, cfg['forecasting_horizon']])
    features = np.zeros([len(sequence)-cfg['sequence_length']-cfg['forecasting_horizon']+1, cfg['sequence_length'], num_features])
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + cfg['sequence_length']
        out_end_ix = end_ix + cfg['forecasting_horizon']
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        if num_features > 1:
            y[i] = seq_y[[cfg['target_feature']]]
            x[i] = seq_x[[cfg['target_feature']]]
            features[i] = seq_x

        else:
            y[i] = seq_y
            x[i] = seq_x

    return x, features, y


def gen_sequence(id_df, seq_length, cols):
    data_matrix = id_df[cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def train_test_split_avocado(df, cfg):
    train_conv, test_conv = [], []
    train_org, test_org = [], []
    holdout_train_conv, holdout_test_conv = [], []
    holdout_train_org, holdout_test_org = [], []
    for county in df["region"].unique():
        for sequence in gen_sequence(
                df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] != 2018, ])],
                cfg['sequence_length'], ['AveragePrice']):
            train_conv.append(sequence) if county != 'Albany' else holdout_train_conv.append(sequence)
        for sequence in gen_sequence(
                df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] != 2018, ])],
                cfg['sequence_length'], ['AveragePrice']):
            train_org.append(sequence) if county != 'Albany' else holdout_train_org.append(sequence)
        for sequence in gen_sequence(
                df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
                cfg['sequence_length'], ['AveragePrice']):
            test_conv.append(sequence) if county != 'Albany' else holdout_test_conv.append(sequence)
        for sequence in gen_sequence(
                df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
                cfg['sequence_length'], ['AveragePrice']):
            test_org.append(sequence) if county != 'Albany' else holdout_test_org.append(sequence)
    train_conv, train_org = np.asarray(train_conv), np.asarray(train_org)
    test_conv, test_org = np.asarray(test_conv), np.asarray(test_org)
    holdout_train_conv, holdout_test_conv = np.asarray(holdout_train_conv), np.asarray(holdout_train_org)
    holdout_train_org, holdout_test_org = np.asarray(holdout_test_conv), np.asarray(holdout_test_org)

    train = np.concatenate([train_conv, train_org], axis=0)
    test = np.concatenate([test_conv, test_org], axis=0)

    return train, test
