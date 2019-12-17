import numpy as np
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Sarimax:
    def __init__(self, df, cfg):
        self.series = df[cfg['target_feature']]
        self.model = SARIMAX(self.series, order=(3, 1, 0), seasonal_order=(0, 0, 0, 12))

    def fit_model(self):
        # Fit model
        self.model = self.model.fit(disp=0)
        print(self.model.summary())

    def plot_autocorrelation(self):
        # Plot auto correlation
        autocorrelation_plot(self.series)
        plt.show()

    def predict_arima(self, series):
        return self.model.predict(series)


class Arima:
    def __init__(self, df, cfg):
        self.series = df[cfg['target_feature']]
        self.model = ARIMA(self.series, order=(3, 1, 0))

    def fit_model(self):
        # Fit model
        self.model = self.model.fit(disp=0)
        print(self.model.summary())

    def plot_autocorrelation(self):
        # Plot auto correlation
        autocorrelation_plot(self.series)
        plt.show()

    def predict_arima(self, series):
        return self.model.predict(series)
