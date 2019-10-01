import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.modeling.walk_forward_prediction import walk_forward_validation
from src.utility.compute_coverage import compute_coverage
from src.processing.split_data import split_sequence, train_test_split
from src.networks.autoencoder import build_model
from src.preparation.load_data import load_data
from src.modeling.config.open_config import load_config_file


# Load config
cfg = load_config_file('config\\config.yml', print_config=True)
# Overload config
cfg['data_source'] = 'm4'
print('Overload config:', cfg)
df = load_data(cfg)
df.dropna(axis=1, how='any', inplace=True)
df = df.iloc[:, 0:10]
coverage_80pi = np.zeros([len(df.columns), cfg['forecasting_horizon']])
coverage_95pi = np.zeros([len(df.columns), cfg['forecasting_horizon']])
i = 0

# Train autoencoder
print(df.shape)
encoder_train, encoder_train_x = list(), list()
scaler = MinMaxScaler(feature_range=[0, 1])
for (columnName, columnData) in df.iteritems():
    train, _ = train_test_split(df[[columnName]].values, cfg['test_size'])
    train_scaled = scaler.fit_transform(train)
    train_x, train_y = split_sequence(train_scaled, cfg)
    encoder_train.append(train_x)
encoder_train = np.array(encoder_train)
encoder_train = encoder_train.reshape((encoder_train.shape[0]*encoder_train.shape[1], encoder_train.shape[2],
                                       encoder_train.shape[3]))

# encoder_train_x = np.array(encoder_train_x)
# encoder_train_x = encoder_train_x.reshape((encoder_train_x.shape[0]*encoder_train_x.shape[1],
# encoder_train_x.shape[2],
#                                           encoder_train_x.shape[3]))
# encoder_train = scaler.fit_transform(encoder_train)
print(encoder_train.shape)
encoder = build_model(encoder_train, cfg)

for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    # print('Column Contents : ', columnData.values)
    df_i = scaler.fit_transform(df[[columnName]].values)
    prediction_sequence, mc_mean, total_uncertainty, quantile_80, quantile_95, test = walk_forward_validation(df_i, scaler, cfg, encoder)
    for j in range(cfg['forecasting_horizon']):
        coverage_80pi[i, j] = compute_coverage(upper_limits=mc_mean[:, j] + 1.28 * total_uncertainty[:, j],
                                               lower_limits=mc_mean[:, j] - 1.28 * total_uncertainty[:, j],
                                               actual_values=test)
        coverage_95pi[i, j] = compute_coverage(upper_limits=mc_mean[:, j] + 1.96 * total_uncertainty[:, j],
                                               lower_limits=mc_mean[:, j] - 1.96 * total_uncertainty[:, j],
                                               actual_values=test)
    # plot_predictions(df_i, mc_mean, [mc_mean - 1.28 * total_uncertainty, mc_mean + 1.28 * total_uncertainty],
    #                 [mc_mean - 1.96 * total_uncertainty, mc_mean + 1.96 * total_uncertainty])
    i += 1
for j in range(cfg['forecasting_horizon']):
    print('80%-prediction interval coverage: ', j, np.mean(coverage_80pi[:, j]))
    print('95%-prediction interval coverage: ', j, np.mean(coverage_95pi[:, j]))
