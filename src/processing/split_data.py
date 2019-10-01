import numpy as np


# Split data into training and testing
def train_test_split(df, test_split=0.2):
    test_size = int(test_split*len(df))
    return df[:-test_size], df[-test_size:]


# split a univariate sequence into samples
def split_sequence(sequence, cfg):
    x = np.zeros([len(sequence), cfg['sequence_length'], 1])
    y = np.zeros([len(sequence), cfg['forecasting_horizon']])
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + cfg['sequence_length']
        out_end_ix = end_ix + cfg['forecasting_horizon']
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        x[i] = seq_x
        y[i] = seq_y
    # y = y.reshape((train_y.shape[0], train_y.shape[1]))
    # return np.array(x), np.array(y)
    return x, y
