import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras import backend as K

from src.modeling.config.open_config import load_config_file
from src.preparation.load_data import load_data
from src.networks.train_model import train_model
from src.modeling.pre_traning_autoencoder import pre_training
from src.utility.compute_coverage import compute_coverage
from src.dataclasses.Avocado import Avocado


def print_coverage(mean, uncertainty, actual_values):
    coverage_80pi = compute_coverage(upper_limits=mean + 1.28 * uncertainty,
                                     lower_limits=mean - 1.28 * uncertainty,
                                     actual_values=actual_values)
    coverage_95pi = compute_coverage(upper_limits=mean + 1.96 * uncertainty,
                                     lower_limits=mean - 1.96 * uncertainty,
                                     actual_values=actual_values)
    print('80%-prediction interval coverage: ', coverage_80pi)
    print('95%-prediction interval coverage: ', coverage_95pi)


def stochastic_dropout(x, f, model, encoder, scaler, r, cfg):
    enc = K.function([encoder.layers[0].input, K.learning_phase()], [encoder.layers[-1].output])
    NN = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    enc_pred = np.vstack(enc([x, r]))
    enc_pred = np.concatenate([enc_pred, f], axis=2)
    trans_pred = scaler.transform(enc_pred.reshape(-1, 128 + 8)).reshape(-1, cfg['sequence_length'], 128 + 8)
    NN_pred = NN([trans_pred, r])

    return np.vstack(NN_pred)


def pipeline(data, cfg, encoder=None):
    # If using an encoder, extract features from training data,
    enc_pred_train = encoder.predict(data.x_train)
    train_x = np.concatenate([enc_pred_train, data.f_train], axis=2)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(-1, 128 + 8)).reshape(-1, cfg['sequence_length'], 128 + 8)
    model = train_model(train_x, data.y_train, cfg)

    mae_test = []
    mse_test = []
    for _ in tqdm.tqdm(range(0, 200)):
        pred = stochastic_dropout(data.x_test, data.f_test, model, encoder, scaler, 0.5, cfg)
        mae_test.append(mean_absolute_error(data.y_test, pred))
        mse_test.append(mean_squared_error(data.y_test, pred))
    print(np.mean(mae_test), np.std(mae_test))

    mae_holdout = []
    mse_holdout = []
    pred_holdout = []
    for _ in tqdm.tqdm(range(0, 200)):
        pred = stochastic_dropout(data.x_holdout_org, data.f_holdout_org, model, encoder, scaler, 0.5, cfg)
        mae_holdout.append(mean_absolute_error(data.y_holdout_org, pred))
        mse_holdout.append(mean_squared_error(data.y_holdout_org, pred))
        pred_holdout.append(pred)
    print(np.mean(mae_holdout), np.std(mae_holdout))

    uncertainty = np.sqrt(np.mean(mse_test) + np.std(np.hstack(pred_holdout).T, axis=0)**2)

    print_coverage(np.mean(mse_holdout), uncertainty, data.y_holdout_org)

    x = np.linspace(1, len(data.y_holdout_org), len(data.y_holdout_org))
    plt.plot(x, np.mean(np.hstack(pred_holdout).T, axis=0), color='orange')
    plt.plot(x, data.y_holdout_org, color='green')
    plt.title('AveragePrice ORGANIC Albany')
    plt.fill_between(x,
                     np.mean(np.hstack(pred_holdout).T, axis=0) - 1.28*uncertainty,
                     np.mean(np.hstack(pred_holdout).T, axis=0) + 1.28*uncertainty,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(x,
                     np.mean(np.hstack(pred_holdout).T, axis=0) - 1.96*uncertainty,
                     np.mean(np.hstack(pred_holdout).T, axis=0) + 1.96*uncertainty,
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()


def main():
    cfg = load_config_file('config\\config.yml', print_config=True)

    df = load_data(cfg)
    #if cfg['differencing']:
    #    df = df.diff(periodes=1)
    print(df.shape)
    cfg['target_feature'] = 'AveragePrice'
    cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
    #df_diff[cols] = df[cols].diff(periods=1, axis=0)
    #df_diff = df_diff.drop(df_diff.index[0])
    print(df)
    avocado_data = Avocado(df, cfg)


    if cfg['autoencoder']:
        encoder, cfg = pre_training(data=avocado_data, cfg=cfg)
    else:
        encoder = None
    pipeline(avocado_data, cfg, encoder)

    print(cfg)


if __name__ == '__main__':
    main()
