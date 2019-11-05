import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_sine_data(xmin=0, xmax=300, num_points=1500, noise=True):
    x_data = np.linspace(xmin, xmax, num_points)
    y_data = np.random.uniform(0, 2)*np.sin(np.random.uniform(0, 2)*x_data)
    if noise:
        y_data += np.random.normal(0, 0.2, num_points)
    return pd.DataFrame({'x': x_data, 'y': y_data})


def generate_coefficients(p):
    filter_stable = False
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while not filter_stable:
        true_theta = np.random.random(p) - 0.5
        coefficients = np.append(1, -true_theta)
        # check if magnitude of all poles is less than one
        if np.max(np.abs(np.roots(coefficients))) < 1:
            filter_stable = True
    return true_theta


def generate_arp_data(p=2, burn_in=600, num_points=2000):
    arp_sequence = np.zeros(p+num_points+burn_in)
    arp_sequence[:p] = np.random.normal(0, 0.2, p)
    coefficients = generate_coefficients(p)
    for i in range(p, num_points+burn_in+p):
        for j in range(p):
            arp_sequence[i] += arp_sequence[i-j-1]*coefficients[j]

    arp_sequence += np.random.normal(0, 0.1, len(arp_sequence))
    return pd.DataFrame({'y': arp_sequence[(p+burn_in):]})


def generate_time_series_data(xmin=0, xmax=300, num_points=500, num_time_series=10):
    time_series_data = pd.DataFrame()
    for i in range(num_time_series):
        sine_data = generate_sine_data(xmin, xmax, num_points, noise=False)
        x = sine_data['x']
        seasonal = sine_data['y']
        trend = x*np.random.uniform(0, 1)/xmax
        if np.random.uniform(0, 1) < 0.5:
            noise = np.random.normal(0, 0.5, num_points)
            time_series_data['y' + str(i)] = seasonal + trend + noise
        else:
            noise = np.random.normal(0, 0.01, num_points)
            time_series_data['y'+str(i)] = seasonal*trend*noise
    return time_series_data


if __name__ == '__main__':
    df = generate_time_series_data()
    plt.figure()
    plt.plot(df)
    plt.show()
