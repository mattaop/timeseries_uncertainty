from sklearn.preprocessing import MinMaxScaler


def scale_data(data, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(data)
