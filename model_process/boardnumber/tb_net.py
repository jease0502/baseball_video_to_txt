import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.applications import resnet
import efficientnet.tfkeras as efn


def CNN():
    model = efn.EfficientNetB0(input_shape=[64, 192, 3], include_top=False)
    inputs = model.input
    x = model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(512, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(1, activation='sigmoid')(x)

    return K.Model(inputs, x)
