from keras.layers.core import Lambda
import keras.backend as K


def permanent_dropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))
