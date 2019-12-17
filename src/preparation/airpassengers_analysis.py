import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from src.preparation.load_data import load_data
from statsmodels.tsa.seasonal import seasonal_decompose


df, cfg = load_data(data_set='AirPassengers')
plot_acf(df.values, lags=30)
plot_pacf(df.values, lags=30)
plt.show()

result = seasonal_decompose(df.values, model='multiplicative', freq=12)
result.plot()
plt.show()


fig = plt.figure()
ax = fig.gca()
ax.plot(df)
plt.xlabel('Year')
plt.ylabel('Number of air passengers (in 1000s)')
plt.title('Air Passengers Data (1949-1960)')
ax.set_xticks(df.index[::12])
ax.set_xticklabels(df.index[::12])
plt.show()

scaler = MinMaxScaler()
df['y'] = np.log(df['y'])
df['y'] = scaler.fit_transform(df['y'].values.reshape(-1, 1))
plot_acf(df, lags=30)
plot_pacf(df, lags=30)
plt.show()

df = df.diff(periods=1)
df = df.diff(periods=1)

#df = df.diff(periods=12)
df.dropna(inplace=True)
#df = df.iloc[13:]


fig = plt.figure()
ax = fig.gca()
ax.plot(df)
plt.xlabel('Year')
plt.ylabel('Scaled and differenced data')
plt.title('Scaled and Differenced Data Air Passengers Data (1949-1960)')
ax.set_xticks(df.index[::12])
ax.set_xticklabels(df.index[::12])
plt.show()

plot_acf(df, lags=30)
plot_pacf(df, lags=30)
plt.show()
