import numpy as np
# from sklearn.model_selection import train_test_split


# Split data into training and testing
def train_test_split(df, test_split=0.2):
    test_size = int(test_split*len(df))
    return df[:-test_size], df[-test_size:]


# split a univariate sequence into samples
def split_sequence(sequence, cfg, num_features=1):
    x = np.zeros([len(sequence)-cfg['sequence_length']-1+1, cfg['sequence_length'], 1])
    y = np.zeros([len(sequence)-cfg['sequence_length']-1+1, 1])
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + cfg['sequence_length']
        out_end_ix = end_ix + 1
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        y[i] = seq_y
        x[i] = seq_x
    return x, y


def split_multiple_sequences(data, cfg):
    x, y = [], []
    for i in range(len(data)):
        x_i, y_i = split_sequence(data[i], cfg)
        x.append(x_i)
        y.append(y_i)
    x = np.asarray(x)
    y = np.asarray(y)

    x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3]])
    y = y.reshape([y.shape[0] * y.shape[1], y.shape[2]])
    return x, y
