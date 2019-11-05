import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from keras.models import *
from keras.layers import *
from keras.layers.core import Lambda
from keras import backend as K
from src.dataclasses.Avocado import Avocado

df = pd.read_csv('C://Users//mathi//PycharmProjects//timeseries_uncertainty//data//raw//avocado.csv', index_col=0)
print(df.shape)
df.head()


def plot_seris(county, typ):
    plt.figure(figsize=(9, 6))
    reg_train = df[np.logical_and(df['type'] == typ, df['year'] != 2018)].groupby('region')
    p_train = reg_train.get_group(county)[['Date', 'AveragePrice']].reset_index(drop=True)
    plt.plot(range(0, len(p_train)), p_train.AveragePrice.values)

    reg_test = df[np.logical_and(df['type'] == typ, df['year'] == 2018)].groupby('region')
    p_test = reg_test.get_group(county)[['Date', 'AveragePrice']].reset_index(drop=True)
    plt.plot(range(len(p_train), len(p_train) + len(p_test)), p_test.AveragePrice.values)
    plt.title('AveragePrice' + ' ' + typ.upper() + ' ' + county)
    plt.show()


plot_seris('NewYork','conventional')


# CREATE GENERATOR FOR LSTM WINDOWS AND LABELS #
sequence_length = 4


def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    return data_matrix[seq_length:num_elements, :]


### CREATE TRAIN/TEST PRICE DATA ###
X_train_c, X_train_o = [], []
X_test_c, X_test_o = [], []
X_other_train_c, X_other_train_o = [], []
X_other_test_c, X_other_test_o = [], []

for county in df["region"].unique():
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] != 2018, ])],
            sequence_length, ['AveragePrice']):
        X_train_c.append(sequence) if county != 'Albany' else X_other_train_c.append(sequence)
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] != 2018, ])],
            sequence_length, ['AveragePrice']):
        X_train_o.append(sequence) if county != 'Albany' else X_other_train_o.append(sequence)

    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        X_test_c.append(sequence) if county != 'Albany' else X_other_test_c.append(sequence)
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        X_test_o.append(sequence) if county != 'Albany' else X_other_test_o.append(sequence)

X_train_c, X_train_o = np.asarray(X_train_c), np.asarray(X_train_o)
X_test_c, X_test_o = np.asarray(X_test_c), np.asarray(X_test_o)
X_other_train_c, X_other_train_o = np.asarray(X_other_train_c), np.asarray(X_other_train_o)
X_other_test_c, X_other_test_o = np.asarray(X_other_test_c), np.asarray(X_other_test_o)

### CREATE TRAIN/TEST LABEL ###
y_train_c, y_train_o = [], []
y_test_c, y_test_o = [], []
y_other_train_c, y_other_train_o = [], []
y_other_test_c, y_other_test_o = [], []

for county in df["region"].unique():
    for sequence in gen_labels(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] != 2018, ])],
            sequence_length, ['AveragePrice']):
        y_train_c.append(sequence) if county != 'Albany' else y_other_train_c.append(sequence)
    for sequence in gen_labels(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] != 2018, ])],
            sequence_length, ['AveragePrice']):
        y_train_o.append(sequence) if county != 'Albany' else y_other_train_o.append(sequence)

    for sequence in gen_labels(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        y_test_c.append(sequence) if county != 'Albany' else y_other_test_c.append(sequence)
    for sequence in gen_labels(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        y_test_o.append(sequence) if county != 'Albany' else y_other_test_o.append(sequence)

y_train_c, y_train_o = np.asarray(y_train_c), np.asarray(y_train_o)
y_test_c, y_test_o = np.asarray(y_test_c), np.asarray(y_test_o)
y_other_train_c, y_other_train_o = np.asarray(y_other_train_c), np.asarray(y_other_train_o)
y_other_test_c, y_other_test_o = np.asarray(y_other_test_c), np.asarray(y_other_test_o)

### CONCATENATE TRAIN/TEST DATA AND LABEL ###
X = np.concatenate([X_train_c,X_train_o,X_test_c,X_test_o],axis=0)
y = np.concatenate([y_train_c,y_train_o,y_test_c,y_test_o],axis=0)

print(X.shape, y.shape)

### CREATE TRAIN/TEST EXTERNAL FEATURES ###
col = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']

f_train_c, f_train_o = [], []
f_test_c, f_test_o = [], []
f_other_train_c, f_other_train_o = [], []
f_other_test_c, f_other_test_o = [], []

for county in df["region"].unique():
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] != 2018, ])],
            sequence_length, col):
        f_train_c.append(sequence) if county != 'Albany' else f_other_train_c.append(sequence)
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] != 2018, ])],
            sequence_length, col):
        f_train_o.append(sequence) if county != 'Albany' else f_other_train_o.append(sequence)

    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
            sequence_length, col):
        f_test_c.append(sequence) if county != 'Albany' else f_other_test_c.append(sequence)
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
            sequence_length, col):
        f_test_o.append(sequence) if county != 'Albany' else f_other_test_o.append(sequence)

f_train_c, f_train_o = np.asarray(f_train_c), np.asarray(f_train_o)
f_test_c, f_test_o = np.asarray(f_test_c), np.asarray(f_test_o)
f_other_train_c, f_other_train_o = np.asarray(f_other_train_c), np.asarray(f_other_train_o)
f_other_test_c, f_other_test_o = np.asarray(f_other_test_c), np.asarray(f_other_test_o)


### CONCATENATE TRAIN/TEST EXTERNAL FEATURES ###
F = np.concatenate([f_train_c,f_train_o,f_test_c,f_test_o],axis=0)

print(F.shape)



### DEFINE LSTM AUTOENCODER ###
inputs_ae = Input(shape=(sequence_length, 1))
encoded_ae = LSTM(128, return_sequences=True, dropout=0.3)(inputs_ae, training=True)
decoded_ae = LSTM(32, return_sequences=True, dropout=0.3)(encoded_ae, training=True)
out_ae = TimeDistributed(Dense(1))(decoded_ae)

sequence_autoencoder = Model(inputs_ae, out_ae)
sequence_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
sequence_autoencoder.summary()


### TRAIN AUTOENCODER ###
print(X[:X_train_c.shape[0]+X_train_o.shape[0]].shape)
sequence_autoencoder.fit(X[:X_train_c.shape[0]+X_train_o.shape[0]],
                         X[:X_train_c.shape[0]+X_train_o.shape[0]], batch_size=16, epochs=100, verbose=2, shuffle=True)


### ENCODE PRICE AND CONCATENATE REGRESSORS ###
encoder = Model(inputs_ae, encoded_ae)
XX = encoder.predict(X)
XXF = np.concatenate([XX, F], axis=2)
print(XXF.shape)

### SPLIT TRAIN TEST ###
X_train1, X_test1 = XXF[:X_train_c.shape[0]+X_train_o.shape[0]], XXF[X_train_c.shape[0]+X_train_o.shape[0]:]
y_train1, y_test1 = y[:y_train_c.shape[0]+y_train_o.shape[0]], y[y_train_c.shape[0]+y_train_o.shape[0]:]


### SCALE DATA ###
scaler1 = StandardScaler()
X_train1 = scaler1.fit_transform(X_train1.reshape(-1,128+8)).reshape(-1,sequence_length,128+8)
X_test1 = scaler1.transform(X_test1.reshape(-1,128+8)).reshape(-1,sequence_length,128+8)


### SPLIT TRAIN TEST ###
inputs1 = Input(shape=(X_train1.shape[1], X_train1.shape[2]))
lstm1 = LSTM(128, return_sequences=True, dropout=0.3)(inputs1, training=True)
lstm1 = LSTM(32, return_sequences=False, dropout=0.3)(lstm1, training=True)
dense1 = Dense(50)(lstm1)
out1 = Dense(1)(dense1)

model1 = Model(inputs1, out1)

model1.compile(loss='mse', optimizer='adam', metrics=['mse'])

### FIT FORECASTER ###
history = model1.fit(X_train1, y_train1, epochs=30, batch_size=128, verbose=2, shuffle=True)


### FUNCTION FOR STOCHASTIC DROPOUT ###
def stoc_drop1(r):
    enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    NN = K.function([model1.layers[0].input, K.learning_phase()], [model1.layers[-1].output])

    enc_pred = np.vstack(enc([X[X_train_c.shape[0] + X_train_o.shape[0]:], r]))
    enc_pred = np.concatenate([enc_pred, F[X_train_c.shape[0] + X_train_o.shape[0]:]], axis=2)
    trans_pred = scaler1.transform(enc_pred.reshape(-1, 128 + 8)).reshape(-1, sequence_length, 128 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred)

### COMPUTE STOCHASTIC DROPOUT ###
scores1 = []
for i in tqdm.tqdm(range(0,100)):
    scores1.append(mean_absolute_error(stoc_drop1(0.5), y_test1))

print(np.mean(scores1), np.std(scores1))

### CONCATENATE REGRESSORS ###
XF = np.concatenate([X, F], axis=2)
print(XF.shape)

### SPLIT TRAIN TEST ###
X_train2, X_test2 = XF[:X_train_c.shape[0]+X_train_o.shape[0]], XF[X_train_c.shape[0]+X_train_o.shape[0]:]
y_train2, y_test2 = y[:y_train_c.shape[0]+y_train_o.shape[0]], y[y_train_c.shape[0]+y_train_o.shape[0]:]

scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2.reshape(-1,1+8)).reshape(-1,sequence_length,1+8)
X_test2 = scaler2.transform(X_test2.reshape(-1,1+8)).reshape(-1,sequence_length,1+8)


### DEFINE LSTM FORECASTER ###
inputs2 = Input(shape=(X_train2.shape[1], X_train2.shape[2]))
lstm2 = LSTM(128, return_sequences=True, dropout=0.3)(inputs2, training=True)
lstm2 = LSTM(32, return_sequences=False, dropout=0.3)(lstm2, training=True)
dense2 = Dense(50)(lstm2)
out2 = Dense(1)(dense2)

model2 = Model(inputs2, out2)

model2.compile(loss='mse', optimizer='adam', metrics=['mse'])

### FIT FORECASTER ###
history = model2.fit(X_train2, y_train2, epochs=30, batch_size=128, verbose=2, shuffle=True)


### FUNCTION FOR STOCHASTIC DROPOUT ###
def stoc_drop2(r):
    NN = K.function([model2.layers[0].input, K.learning_phase()], [model2.layers[-1].output])

    trans_pred = scaler2.transform(XF[X_train_c.shape[0] + X_train_o.shape[0]:].reshape(-1, 1 + 8)).reshape(-1,
                                                                                                            sequence_length,
                                                                                                            1 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred)


### COMPUTE STOCHASTIC DROPOUT ###
scores2 = []
for i in tqdm.tqdm(range(0,100)):
    scores2.append(mean_absolute_error(stoc_drop2(0.5), y_test2))

print(np.mean(scores2), np.std(scores2))


### FUNCTION TO GET TEST DATA FOR COUNTY ###
def test_county(county):
    test_X_c, test_X_o = [], []
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        test_X_c.append(sequence)
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        test_X_o.append(sequence)
    test_X_c, test_X_o = np.asarray(test_X_c), np.asarray(test_X_o)

    test_y_c, test_y_o = [], []
    for sequence in gen_labels(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        test_y_c.append(sequence)
    for sequence in gen_labels(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
            sequence_length, ['AveragePrice']):
        test_y_o.append(sequence)
    test_y_c, test_y_o = np.asarray(test_y_c), np.asarray(test_y_o)

    test_F_c, test_F_o = [], []
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "conventional", df["year"] == 2018, ])],
            sequence_length, col):
        test_F_c.append(sequence)
    for sequence in gen_sequence(
            df[np.logical_and.reduce([df["region"] == county, df["type"] == "organic", df["year"] == 2018, ])],
            sequence_length, col):
        test_F_o.append(sequence)
    test_F_c, test_F_o = np.asarray(test_F_c), np.asarray(test_F_o)

    X = np.concatenate([test_X_c, test_X_o], axis=0)
    y = np.concatenate([test_y_c, test_y_o], axis=0)
    F = np.concatenate([test_F_c, test_F_o], axis=0)

    return X, y, F


### FUNCTION FOR STOCHASTIC DROPOUT FOR SINGLE COUNTY ###
def test_stoc_drop1(county, r):
    X, y, F = test_county(county)

    enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    NN = K.function([model1.layers[0].input, K.learning_phase()], [model1.layers[-1].output])

    enc_pred = np.vstack(enc([X, r]))
    enc_pred = np.concatenate([enc_pred, F], axis=2)
    trans_pred = scaler1.transform(enc_pred.reshape(-1, 128 + 8)).reshape(-1, sequence_length, 128 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred), y


def test_stoc_drop2(county, r):
    X, y, F = test_county(county)
    XF = np.concatenate([X, F], axis=2)

    NN = K.function([model2.layers[0].input, K.learning_phase()], [model2.layers[-1].output])

    NN_pred = NN([scaler2.transform(XF.reshape(-1, 1 + 8)).reshape(-1, sequence_length, 1 + 8), r])

    return np.vstack(NN_pred), y

pred1_test, y1_test = test_stoc_drop1('TotalUS', 0.5)


### COMPUTE STOCHASTIC DROPOUT FOR SINGLE COUNTY ###
mae1_test = []
for i in tqdm.tqdm(range(0,100)):
    mae1_test.append(mean_absolute_error(test_stoc_drop1('TotalUS', 0.5)[0], y1_test))

print(np.mean(mae1_test), np.std(mae1_test))

pred2_test, y2_test = test_stoc_drop2('TotalUS', 0.5)

### COMPUTE STOCHASTIC DROPOUT FOR SINGLE COUNTY ###
mae2_test = []
for i in tqdm.tqdm(range(0,100)):
    mae2_test.append(mean_absolute_error(test_stoc_drop2('TotalUS', 0.5)[0], y2_test))

print(np.mean(mae2_test), np.std(mae2_test))


### PLOT AVG AND UNCERTAINTY OF RESULTS ###
bar = plt.bar([0,1],[np.mean(mae1_test), np.mean(mae2_test)], yerr=[2.95*np.std(mae1_test), 2.95*np.std(mae2_test)])
plt.xticks([0,1], ['model1','model2'], rotation=90)
bar[0].set_color('cyan'), bar[1].set_color('magenta')
plt.title('TotalUS')
plt.show()


### FUNCTION FOR STOCHASTIC DROPOUT FOR UNSEEN COUNTY ###
def test_other_drop1(r, typ):
    avocado_set = Avocado(df, cfg={'model': 'lstm', 'data_source': 'avocado', 'number_of_nodes': 50, 'number_of_epochs': 100, 'number_of_mc_forward_passes': 100, 'batch_size': 128, 'dropout_rate_test': 0.5, 'patience': 10, 'sequence_length': 4, 'forecasting_horizon': 1, 'test_size': 0.4, 'multi_step_prediction': False, 'differencing': False, 'mc_dropout': True, 'autoencoder': 'lstm', 'load_weights_autoencoder': False})

    if typ == 'conventional':
        X = np.concatenate([X_other_train_c, X_other_test_c], axis=0)
        F = np.concatenate([f_other_train_c, f_other_test_c], axis=0)
        y = np.concatenate([y_other_train_c, y_other_test_c], axis=0)
        print(np.array_equal(X, avocado_set.x_holdout_conv))
        print(np.array_equal(y, avocado_set.y_holdout_conv))
        print(np.array_equal(F, avocado_set.f_holdout_conv))
    elif typ == 'organic':
        X = np.concatenate([X_other_train_o, X_other_test_o], axis=0)
        F = np.concatenate([f_other_train_o, f_other_test_o], axis=0)
        y = np.concatenate([y_other_train_o, y_other_test_o], axis=0)
        print(np.array_equal(X, avocado_set.x_holdout_org))
        print(np.array_equal(y, avocado_set.y_holdout_org))
        print(np.array_equal(F, avocado_set.f_holdout_org))

    enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    NN = K.function([model1.layers[0].input, K.learning_phase()], [model1.layers[-1].output])

    enc_pred = np.vstack(enc([X, r]))
    enc_pred = np.concatenate([enc_pred, F], axis=2)
    trans_pred = scaler1.transform(enc_pred.reshape(-1, 128 + 8)).reshape(-1, sequence_length, 128 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred), y


def test_other_drop2(r, typ):
    if typ == 'conventional':
        X = np.concatenate([X_other_train_c, X_other_test_c], axis=0)
        F = np.concatenate([f_other_train_c, f_other_test_c], axis=0)
        y = np.concatenate([y_other_train_c, y_other_test_c], axis=0)
    elif typ == 'organic':
        X = np.concatenate([X_other_train_o, X_other_test_o], axis=0)
        F = np.concatenate([f_other_train_o, f_other_test_o], axis=0)
        y = np.concatenate([y_other_train_o, y_other_test_o], axis=0)

    NN = K.function([model2.layers[0].input, K.learning_phase()], [model2.layers[-1].output])

    trans_pred = scaler2.transform(np.concatenate([X, F], axis=2).reshape(-1, 1 + 8)).reshape(-1, sequence_length,
                                                                                              1 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred), y

### COMPUTE STOCHASTIC DROPOUT FOR UNSEEN COUNTY ###
mae1_other, p1_other = [], []
for i in tqdm.tqdm(range(0,100)):
    pred1_other, true1_other = test_other_drop1(0.5, 'organic')
    mae1_other.append(mean_absolute_error(pred1_other, true1_other))
    p1_other.append(pred1_other)

print(np.mean(mae1_other), np.std(mae1_other))

### COMPUTE STOCHASTIC DROPOUT FOR UNSEEN COUNTY ###
mae2_other, p2_other = [], []
for i in tqdm.tqdm(range(0,100)):
    pred2_other, true2_other = test_other_drop2(0.5, 'organic')
    mae2_other.append(mean_absolute_error(pred2_other, true2_other))
    p2_other.append(pred2_other)

print(np.mean(mae2_other), np.std(mae2_other))

### PLOT AVG AND UNCERTAINTY OF RESULTS ###
bar = plt.bar([0,1],[np.mean(mae1_other), np.mean(mae2_other)], yerr=[2.95*np.std(mae1_other), 2.95*np.std(mae2_other)])
plt.xticks([0,1], ['model1','model2'], rotation=90)
bar[0].set_color('cyan'), bar[1].set_color('magenta')
plt.title('ORGANIC Albany')
plt.show()

plt.plot(np.mean(np.hstack(p1_other).T,axis=0),color='orange')
plt.plot(pred1_other,color='green')
plt.title('AveragePrice CONVENTIONAL Albany - Model1')
plt.show()

plt.plot(np.mean(np.hstack(p2_other).T,axis=0),color='orange')
plt.plot(pred2_other,color='green')
plt.title('AveragePrice CONVENTIONAL Albany - Model2')
plt.show()
