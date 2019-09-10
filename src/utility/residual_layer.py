from keras.layers import add


def residual_layer(inputs, layer):
    x = layer(inputs)
    x = add([x, inputs])
    return x
