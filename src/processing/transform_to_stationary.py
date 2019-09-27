import numpy as np


def difference_transformation(data, interval=1):
    difference = list()
    for i in range(interval, len(data)):
        difference.append(data[i]-data[i-interval])
    return difference


def inverse_difference_transformation(data, predictions, interval=1):
    reverse = np.empty_like(predictions)
    print(len(np.shape(predictions)))
    if len(np.shape(predictions)) > 1:
        reverse[:, 0] = predictions[:, 0] + data[-interval-(len(predictions[0]))]
        for i in range(interval, len(predictions[0])):
            reverse[:, i] = predictions[:, i] + reverse[:, i-interval]
    else:
        reverse[0] = predictions[0] + data[-interval - (len(predictions))]
        for i in range(interval, len(predictions)):
            reverse[i] = predictions[i] + reverse[i - interval]
    return reverse

