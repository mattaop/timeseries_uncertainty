import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_avocado_data(df, cfg):
    df['Date'] = pd.to_datetime(df['Date'])
    cfg['target_feature'] = 'AveragePrice'
    # df = df.pivot_table(values='AveragePrice', index='Date', columns=['region', 'type'], aggfunc='mean')
    df = df.pivot_table(index='Date', columns=['region', 'type'], aggfunc='mean')
    df = df.fillna(method='backfill').dropna()
    # albany_conventional = extract_external_features(df, region='Albany', avocado_type='conventional')
    cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags',
            'XLarge Bags']
    df = df[cols]
    cfg['num_features'] = len(cols)
    return df, cfg


def extract_external_features(df, region, avocado_type):
    external_features = pd.DataFrame()
    for feature in df.columns.get_level_values(0):
        external_features[feature] = df[[(feature, region, avocado_type)]].values.reshape(len(df[[(feature, region, avocado_type)]]))
    return external_features
