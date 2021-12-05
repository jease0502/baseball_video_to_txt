from re import S
import cv2
import numpy as np
import tensorflow as tf


class Bso(object):
    def __init__(self, model_path1, model_path2):
        self.bso_dict = {0: "B", 1: "S", 2: "O"}
        self.model_path1 = model_path1
        self.model_path2 = model_path2
        self.load_model()

    def load_model(self):
        self.bso_model = tf.keras.models.load_model(self.model_path1)
        self.bso_number_model = tf.keras.models.load_model(self.model_path2)

    def predict(self, img):
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)
        y_pred = self.bso_model.predict_classes(img)
        number = self.bso_number_model.predict_classes(img)

        return [self.bso_dict[y_pred[0]], number[0]]
