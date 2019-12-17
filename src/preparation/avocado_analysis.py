import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

from src.preparation.load_data import load_data
from src.dataclasses.Avocado import Avocado

df, cfg = load_data()

"""
data = df

label = LabelEncoder()
dicts = {}
label.fit(data.type.drop_duplicates())
dicts['type'] = list(label.classes_)
data.type = label.transform(data.type)
cols = ['AveragePrice', 'type', 'year', 'Total Volume', 'Total Bags']
cm = np.corrcoef(data[cols].values.T)
sb.set(font_scale=1.7)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
#plt.show()
"""

#print(df)
#print(df['region'].unique())
#print(len(df['region'].unique()))

series = df.loc[(df['region'] == 'Albany') & (df["type"] == 'conventional')]
#print(series)
series.index = pd.to_datetime(series['Date'])

series_1 = series['AveragePrice']
series_1 = series_1.sort_index()
log_data = np.log(series_1)
log_data = log_data.replace(-np.inf, 0)
# Differentiate data to remove change in mean
log_diff_data = log_data.diff(periods=1)

plt.plot(log_diff_data)

#scaler = StandardScaler()
#series['AveragePrice'] = scaler.fit_transform(series[['AveragePrice']])
series = series['AveragePrice']
series = series.sort_index()
#print(series)

result = seasonal_decompose(series.values, model='multiplicative', freq=7)
result.plot()
# plt.show()


plot_acf(series.values, lags=60)
# plt.show()

avocado_data = Avocado(cfg)
# print(avocado_data.y_holdout_org)

print(avocado_data.data.loc[:, pd.IndexSlice['AveragePrice', :, 'conventional']])
print(avocado_data.data.loc[:, pd.IndexSlice['AveragePrice', :, 'conventional']].drop(columns=[('AveragePrice', 'TotalUS', 'conventional')], axis=1).mean(axis=1))
print(avocado_data.data.loc[:, pd.IndexSlice['AveragePrice', 'TotalUS', 'conventional']])
