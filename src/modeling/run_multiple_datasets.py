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
coverage_80pi = np.zeros(len(df.columns))
coverage_95pi = np.zeros(len(df.columns))
i = 0
for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    # print('Column Contents : ', columnData.values)
    df_i = scale_data(df[[columnName]].values)
    prediction_sequence, mc_mean, total_uncertainty, quantile_80, quantile_95, test = walk_forward_validation(df_i, cfg)
    coverage_80pi[i] = compute_coverage(upper_limits=mc_mean + 1.28 * total_uncertainty,
                                        lower_limits=mc_mean - 1.28 * total_uncertainty,
                                        actual_values=test)
    coverage_95pi[i] = compute_coverage(upper_limits=mc_mean + 1.96 * total_uncertainty,
                                        lower_limits=mc_mean - 1.96 * total_uncertainty,
                                        actual_values=test)
    plot_predictions(df_i, mc_mean, [mc_mean - 1.28 * total_uncertainty, mc_mean + 1.28 * total_uncertainty],
                     [mc_mean - 1.96 * total_uncertainty, mc_mean + 1.96 * total_uncertainty])
    i += 1
print('80%-prediction interval coverage: ', np.mean(coverage_80pi))
print('95%-prediction interval coverage: ', np.mean(coverage_95pi))
