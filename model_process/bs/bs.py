import tensorflow as tf
import tensorflow.keras as K
from cv2 import cv2
from tensorflow.python.ops.gen_math_ops import mod


class Bs(object):
    def __init__(self, model_path):
        self.shape = (50, 80)
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model = K.models.load_model(self.model_path)

    def parse_result(self, result):
        pred1, pred2 = result
        num_1 = tf.argmax(pred1, axis=1)
        num_2 = tf.argmax(pred2, axis=1)

        num_1 = num_1.numpy()[0]
        num_2 = num_2.numpy()[0]

        return num_1, num_2

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.shape[::-1])

        img = tf.convert_to_tensor(img, tf.float32)
        img /= 255.
        img = tf.expand_dims(img, axis=0)

        result = self.model(img, training=False)
        num_1, num_2 = self.parse_result(result)
        return str(num_1) + "-" + str(num_2)
