def main():
    cfg = load_config_file('config\\config.yml', print_config=True)

    df = load_data(cfg)
    scaler = StandardScaler()
    df, cfg = process_avocado_data(df, cfg)
    df[df.columns.values] = scaler.fit_transform(df[df.columns.values])
    if cfg['differencing']:
        df_difference = difference_transformation(df)
        # df_difference = scaler.fit_transform(df_difference)

        if cfg['autoencoder']:
            encoder = train_autoencoder(df_difference, cfg)
        else:
            encoder = None
        predictions_difference, mc_mean_difference,  mc_median_difference, total_uncertainty_difference, difference_quantile_80, difference_quantile_95, _ = walk_forward_validation(df_difference, cfg, encoder)
        plot_predictions(df_difference, mc_mean_difference, mc_median_difference,
                         np.array([mc_mean_difference - 1.28 * total_uncertainty_difference,
                                   mc_mean_difference + 1.28 * total_uncertainty_difference]),
                         np.array([mc_mean_difference - 1.96 * total_uncertainty_difference,
                                   mc_mean_difference + 1.96 * total_uncertainty_difference]), cfg)
        # predictions = inverse_difference_transformation(df, predictions_difference)
        mc_mean = inverse_difference_transformation(df, mc_mean_difference)
        mc_median = inverse_difference_transformation(df, mc_median_difference)
        total_uncertainty = total_uncertainty_difference
        # total_uncertainty = inverse_difference_transformation(df, total_uncertainty_difference)
        # quantile_80 = [inverse_difference_transformation(df, difference_quantile_80[0]),
        # inverse_difference_transformation(df, difference_quantile_80[1])]
        # quantile_95 = [inverse_difference_transformation(df, difference_quantile_95[0]),
        # inverse_difference_transformation(df, difference_quantile_95[1])]
        _, test = train_test_split(df, test_split=len(mc_mean[:, 0])/len(df))
        coverage_80pi = compute_coverage(upper_limits=mc_mean[:, 0] + 1.28 * total_uncertainty[:, 0],
                                         lower_limits=mc_mean[:, 0] - 1.28 * total_uncertainty[:, 0],
                                         actual_values=test)
        coverage_95pi = compute_coverage(upper_limits=mc_mean[:, 0] + 1.96 * total_uncertainty[:, 0],
                                         lower_limits=mc_mean[:, 0] - 1.96 * total_uncertainty[:, 0],
                                         actual_values=test)
        print('80%-prediction interval coverage: ', 0, coverage_80pi)
        print('95%-prediction interval coverage: ', 0, coverage_95pi)

    else:
        if cfg['autoencoder']:
            # encoder = train_autoencoder(df_encoder, cfg)
            # encoder.summary()
            encoder = pre_training(df[cfg['target_feature']].drop(columns=[('Albany', 'conventional')]), cfg)
        else:
            encoder = None
        predictions, mc_mean, mc_median, total_uncertainty, quantile_80, quantile_95, _ = walk_forward_validation(df, cfg, encoder)

    # plot_predictions(df, mc_mean, quantile_80, quantile_95, cfg)
    plot_predictions(df[['AveragePrice']], mc_mean, mc_median,
                     np.array([mc_mean - 1.28*total_uncertainty, mc_mean + 1.28*total_uncertainty]),
                     np.array([mc_mean - 1.96*total_uncertainty, mc_mean + 1.96*total_uncertainty]), cfg)
    print(cfg)


if __name__ == '__main__':
    main()
