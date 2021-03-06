import numpy as np

from src.modeling.single_time_series import walk_forward_validation
from src.utility.compute_coverage import compute_coverage
from src.preparation.load_data import load_data
from src.preparation.config.open_config import load_config_file
from src.modeling.pre_traning_autoencoder import pre_training


def main():
    # Load data
    df, cfg = load_data()
    print(df.shape)
    # Initialize lists
    coverage_80pi = np.zeros([len(df.columns), cfg['forecasting_horizon']])
    coverage_95pi = np.zeros([len(df.columns), cfg['forecasting_horizon']])
    i = 0
    print(df)
    # Pre train autoencoder
    # encoder, scaler = pre_training(df, cfg)

    # Train over all time series in df
    for (columnName, columnData) in df.iteritems():
        print('Column Name : ', columnName)
        # print('Column Contents : ', columnData.values)
        df_i = scaler.fit_transform(df[[columnName]].values)
        prediction_sequence, mc_mean, mc_median, total_uncertainty, quantile_80, quantile_95, test = walk_forward_validation(df_i, cfg)
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

    # Print coverage for each forecasting horizon
    for j in range(cfg['forecasting_horizon']):
        print('Mean intervals over', len(df.columns.values), 'data sets')
        print('80%-prediction interval coverage: ', j, np.mean(coverage_80pi[:, j]))
        print('95%-prediction interval coverage: ', j, np.mean(coverage_95pi[:, j]))


if __name__ == '__main__':
    main()
