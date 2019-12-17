import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from src.preparation.load_data import load_data

df, cfg = load_data(data_set='oslo_temperature')
print(df)

plot_acf(df.values, lags=30)
plot_pacf(df.values, lags=30)
plt.show()

result = seasonal_decompose(df.values, model='add', freq=12)
result.plot()
plt.show()

fig = plt.figure()
ax = fig.gca()
ax.plot(df)
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Temperature of Oslo')
ax.set_xticks(df.index[::10*12].date)
ax.set_xticklabels(df.index[::10*12].date)
plt.show()


scaler = MinMaxScaler()
