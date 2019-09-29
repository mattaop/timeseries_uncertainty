from src.modeling.walk_forward_prediction import walk_forward_validation, compute_coverage, plot_predictions
from src.preparation.load_data import load_data
from src.modeling.config.open_config import load_config_file
from src.processing.scale import scale_data
import numpy as np


# Load config
cfg = load_config_file('config\\config.yml', print_config=True)
# Overload config
cfg['data_source'] = 'm4'
print('Overload config:', cfg)
df = load_data(cfg)
df.dropna(axis=1, how='any', inplace=True)
df = df.iloc[:, 0:100]
coverage_80pi = np.zeros([len(df.columns), cfg['forecasting_horizon']])
coverage_95pi = np.zeros([len(df.columns), cfg['forecasting_horizon']])
i = 0
print(df.columns.values)
for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    # print('Column Contents : ', columnData.values)
    df_i = scale_data(df[[columnName]].values)
    prediction_sequence, mc_mean, total_uncertainty, quantile_80, quantile_95, test = walk_forward_validation(df_i, cfg)
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
