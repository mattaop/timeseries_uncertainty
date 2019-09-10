import numpy as np
import pandas as pd


def generate_sine_data(xmin=0, xmax=300, num_points=1500):
    x_data = np.linspace(xmin, xmax, num_points)
    y_data = np.sin(x_data) + np.random.normal(0, 0.1, num_points)
    return pd.DataFrame({'x': x_data, 'y': y_data})

