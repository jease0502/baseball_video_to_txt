import tensorflow as tf
import tensorflow.keras as K
import efficientnet.tfkeras as efn


def CRNN(num_classes):
    model = efn.EfficientNetB0(input_shape=[128, 400, 3], include_top=False)
    x = model.output
    x = K.layers.Reshape((-1, x.shape[-1]))(x)

    x = K.layers.Bidirectional(K.layers.LSTM(
        units=256, return_sequences=True))(x)
    x = K.layers.Bidirectional(K.layers.LSTM(
        units=256, return_sequences=True))(x)
    x = K.layers.Dense(units=num_classes)(x)

    return K.Model(model.input, x)
