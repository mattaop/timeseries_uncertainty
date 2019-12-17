import numpy as np
from sklearn.metrics import mean_squared_error

from src.utility.compute_coverage import print_coverage
from src.utility.plot_forecast import plot_forecast


def last_day_prediction(x, y):
    pred = x[:, -1, 0]
    mse = mean_squared_error(y, pred)
    return mse, pred


def exponential_smoothing(x, y):
    ts = x[:, -1, 0]
    s = []
    alpha = 0.5
    s.append(ts[0])
    for i in range(len(ts)):
        s.append(alpha*ts[i] + (1-alpha)*s[i])
    mse = F(y, s[1:])
    return mse, s[1:]


def statistics(x, y, mse, pred):
    # Naive benchmarks
    mse_last_day, last_day = last_day_prediction(x, y)
    # mse_es = mean_squared_error(y, pred_es)
    mse_es, pred_es = exponential_smoothing(x, y)
    mse_zero = mean_squared_error(y, np.zeros_like(y))

    # Compute mean, uncertainty and noise of Monte Carlo forecasts
    mean = np.mean(np.hstack(pred).T, axis=0)
    mc_uncertainty = np.std(np.hstack(pred).T, axis=0)
    inherent_noise = np.sqrt(np.mean(mse))
    uncertainty = np.sqrt(inherent_noise ** 2 + mc_uncertainty ** 2)

    # Compute coverage of forecasts
    print_coverage(mean, uncertainty, y)
    print("_______________________________________")
    print("|Method                     | MSE     |")
    print("|Diff last day prediction   |", "%.5f" % mse_zero, "|")
    print("|Last day prediction        |", "%.5f" % mse_last_day, "|")
    print("|Exponential smoothing      |", "%.5f" % mse_es, "|")
    print("|LSTM                       |", "%.5f" % np.mean(mse), "|")
    print("_______________________________________")

    plot_forecast(mean, uncertainty, y, last_day, pred_es)
