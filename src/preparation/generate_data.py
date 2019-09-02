import numpy as np
import pandas as pd


def generate_sine_data(xmin=0, xmax=10, num_points=250):
    x_data = np.linspace(xmin, xmax, num_points)
    y_data = np.sin(x_data)
    return pd.DataFrame({'x': x_data, 'y': y_data})

