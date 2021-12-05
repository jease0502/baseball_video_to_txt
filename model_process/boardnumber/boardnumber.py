import numpy as np
from tensorflow import keras
import tensorflow as tf
from cv2 import cv2

from .crnn_net import CRNN
from .tb_net import CNN
from .utilsbs import (
    resize_pad,
    parse_output,
    postprocess_remove_blank
)


class BoardNumber(object):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        self.load_model()

    def load_model(self):
        self.crnn_model = CRNN(11)
        self.crnn_model.load_weights(self.model1)
        self.tb_model = CNN()
        self.tb_model.load_weights(self.model2)

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n = self.predict_number(img)
        tb = self.predict_tb(img)
        tb = ' top' if tb < 0.5 else ' bottom'
        return str(n)+tb

    def predict_number(self, img):

        img = resize_pad(img, [128, 400, 3])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)
        output = self.crnn_model(img, training=False)
        output = parse_output(output)
        result = postprocess_remove_blank(output)
        return result[0]

    def predict_tb(self, img):
        img = resize_pad(img, [64, 192, 3])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)

        output = self.tb_model(img, training=False)
        return output[0][0]
