import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import keras.backend as K
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, concatenate, Dropout, LSTM, Flatten
from keras.layers.core import Lambda
from src.utility.concrete_dropout import ConcreteDropout
from src.preparation.generate_data import generate_sine_data
from src.preparation.load_data import load_raw_data


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


def model_fit(train, config):
    # unpack config
    seq_len, n_nodes, n_epochs, n_batch = config
    # prepare data
    train_x, train_y = split_sequence(train, n_steps=config[0])
    # train_x = train_x.reshape(train_x.shape[0], train_x.shape[1])
    # define model
    inp = Input(shape=(seq_len, 1))
    x = LSTM(n_nodes, activation='relu')(inp)
    # x = LSTM(n_nodes, activation='relu', return_sequences=True)(inp)
    # x = LSTM(n_nodes, activation='relu')(x)
    # x = Flatten()(inp)
    # x = Dense(n_nodes, input_dim=3, activation='relu')(x)
    x = PermaDropout(0.4)(x)
    x = Dense(1)(x)
    model = Model(inp, x)

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    hist = model.fit(train_x, train_y, nb_epoch=n_epochs, batch_size=n_batch, callbacks=[], validation_split=0.1,
                     verbose=2)
    return model


# forecast with a pre-fit model
def forecast(model, history, config):
    # unpack config
    n_input, _, _, _ = config
    # prepare data

    x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def forecast_with_uncertainty(model, history, config):
    # unpack config
    n_input, _, _, _ = config
    # Number of simulations to get
    n_simulations = 30
    result = np.zeros(n_simulations)
    # prepare data
    x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    for i in range(n_simulations):
        result[i] = model.predict(x_input, verbose=0)
    yhat = result.mean()
    yhat_std = result.std()
    return yhat, yhat_std


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


# walk-forward validation for univariate data
def one_step_walk_forward_validation(data, n_test, cfg):
    train, test = train_test_split(data, n_test)
    predictions = np.zeros(len(test))
    standard_deviation = np.zeros(len(test))
    model = model_fit(train, cfg)
    history = [x for x in train]
    # history = train
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat, yhat_std = forecast_with_uncertainty(model, history, cfg)
        # store forecast in list of predictions
        predictions[i] = yhat
        standard_deviation[i] = yhat_std
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)

    # plot predictions
    x_data = np.linspace(1, len(data), len(data))
    x_predictions = np.linspace(len(data) - len(predictions)+1, len(data), len(predictions))
    plt.figure()
    plt.plot(x_data, data, label='Data')
    plt.plot(x_predictions, predictions, label='Predictions')
    plt.fill_between(x_predictions, predictions - standard_deviation, predictions + standard_deviation,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.show()
    return error


# walk-forward validation for univariate data
def multi_step_walk_forward_validation(data, n_test, cfg):
    train, test = train_test_split(data, n_test)
    predictions = np.zeros(len(test))
    standard_deviation = np.zeros(len(test))
    model = model_fit(train, cfg)
    history = [x for x in train]
    # history = train
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat, yhat_std = forecast_with_uncertainty(model, history, cfg)
        # store forecast in list of predictions
        predictions[i] = yhat
        standard_deviation[i] = yhat_std
        # add actual observation to history for the next loop
        history.append(yhat)
    # estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)

    # plot predictions
    x_data = np.linspace(1, len(data), len(data))
    x_predictions = np.linspace(len(data) - len(predictions)+1, len(data), len(predictions))
    plt.figure()
    plt.plot(x_data, data, label='Data')
    plt.plot(x_predictions, predictions, label='Predictions')
    plt.fill_between(x_predictions, predictions - standard_deviation, predictions + standard_deviation,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.show()
    return error


# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
    # fit and evaluate the model n times
    # scores = [one_step_walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
    scores = [multi_step_walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
    return scores


def summarize_scores(name, scores):
    # print a summary
    scores_m, score_std = np.mean(scores), np.std(scores)
    print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
    # box and whisker plot
    plt.boxplot(scores)
    plt.show()


df = generate_sine_data()
# df = load_raw_data()
df.dropna(axis=1, how='all', inplace=True)
# plt.plot(df[['x']], df[['y']])
# plt.show()

df = df[['y']].values

# df = df[['V3']].values
# data split
n_test = 100
# define config
config = [24, 64, 200, 64]  # seq_len, n_nodes, n_epochs, n_batch
# grid search
scores = repeat_evaluate(df, config, n_test, n_repeats=1)
# summarize scores
summarize_scores('mlp', scores)