import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.preparation.load_data import load_data
from src.networks import autoencoder, cnn, lstm, resnet


def train_test_split(df, cfg):
    test_size = int(cfg['test_size']*len(df))
    return df[:-test_size], df[-test_size:]


# split a univariate sequence into samples
def split_sequence(sequence, cfg):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + cfg['sequence_length']
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    return X, y


# forecast with a pre-fit model
def forecast(model, history, cfg):
    # unpack config
    # prepare data

    x_input = np.array(history[-cfg['sequence_length']:])\
        .reshape((1, cfg['sequence_length'], 1))

    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def compute_coverage(upper_limits, lower_limits, actual_values):
    coverage = 0
    for i in range(len(actual_values)):
        if lower_limits[i] < actual_values[i] < upper_limits[i]:
            coverage += 1
    return coverage/len(actual_values)


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def train_model(train_x, train_y, cfg):
    if cfg['model'].lower() == 'resnet':
        model = resnet.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'cnn':
        model = cnn.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'lstm':
        model = lstm.build_model(train_x, train_y, cfg)
    elif cfg['model'].lower() == 'autoencoder':
        model = autoencoder.build_model(train_x, train_y, cfg)
    else:
        ModuleNotFoundError('Model', cfg['model'], 'does not exist')
        model = None
    return model


# walk-forward validation for univariate data
def walk_forward_validation(df, cfg, multi_step_predictions=False):
    train, test = train_test_split(df, cfg)
    train_x, train_y = split_sequence(train, cfg)
    model = train_model(train_x, train_y, cfg)
    num_simulations = 100
    # history = train
    # step over each time-step in the test set
    prediction_sequence = np.zeros([num_simulations, len(test)])
    for j in range(num_simulations):
        history = [x for x in train]
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = forecast(model, history, cfg)

            # store forecast in list of predictions
            if multi_step_predictions:
                history.append(yhat)
            else:
                history.append(test[i])
            prediction_sequence[j, i] = yhat
    mean_predictions = prediction_sequence.mean(axis=0)
    standard_deviation = prediction_sequence.std(axis=0)
    # add actual observation to history for the next loop
    # estimate prediction error
    error = measure_rmse(test, mean_predictions)
    print(' > %.3f' % error)
    coverage_80pi = compute_coverage(upper_limits=mean_predictions + standard_deviation * 1.28,
                                     lower_limits=mean_predictions - standard_deviation * 1.28,
                                     actual_values=test)
    coverage_95pi = compute_coverage(upper_limits=mean_predictions + standard_deviation * 1.96,
                                     lower_limits=mean_predictions - standard_deviation * 1.96,
                                     actual_values=test)

    print('80%-prediction interval coverage: ', coverage_80pi)
    print('95%-prediction interval coverage: ', coverage_95pi)
    plot_predictions(df, mean_predictions, standard_deviation)
    return error, coverage_80pi, coverage_95pi


def plot_predictions(df, mean_predictions, standard_deviation):
    x_data = np.linspace(1, len(df), len(df))
    x_predictions = np.linspace(len(df) - len(mean_predictions) + 1, len(df), len(mean_predictions))
    plt.figure()
    plt.title("Time Series Forecasting Multi-step")
    plt.plot(x_data, df, label='Data')
    plt.plot(x_predictions, mean_predictions, label='Predictions')
    plt.fill_between(x_predictions, mean_predictions - standard_deviation * 1.28,
                     mean_predictions + standard_deviation * 1.28,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(x_predictions, mean_predictions - standard_deviation * 1.96,
                     mean_predictions + standard_deviation * 1.96,
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()


def evaluate_model(df, cfg):
    # fit and evaluate the model
    # one_step_walk_forward_validation(data, n_test, config)
    walk_forward_validation(df, cfg, multi_step_predictions=True)


def main():

    df, cfg = load_data()
    scaler = MinMaxScaler((0, 1))
    df['#Passengers'] = scaler.fit_transform(df['#Passengers'].values.reshape(-1, 1))

    evaluate_model(df['#Passengers'].values.reshape(-1, 1), cfg)


if __name__ == '__main__':
    main()