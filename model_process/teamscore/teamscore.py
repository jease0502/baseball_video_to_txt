import tensorflow as tf
from cv2 import cv2
import numpy as np
import tensorflow.keras as K
import efficientnet.tfkeras as efn
from .parameter import Parameter as params


class Team_score(object):
    def __init__(self, model_path):
        chars = '0123456789'
        self.num_to_char = {i: c for i, c in enumerate(chars)}
        self.model_path = model_path
        self.load_model()

    def CRNN(self, num_classes):
        model = efn.EfficientNetB0(
            input_shape=(128, 400, 3), include_top=False)
        x = model.output

        x = K.layers.Reshape((-1, x.shape[-1]))(x)

        x = K.layers.Bidirectional(K.layers.LSTM(
            units=256, return_sequences=True))(x)
        x = K.layers.Bidirectional(K.layers.LSTM(
            units=256, return_sequences=True))(x)
        x = K.layers.Dense(units=num_classes)(x)

        return K.Model(model.input, x)

    def load_model(self):
        self.model = self.CRNN(11)
        self.model.load_weights(self.model_path)

    def parse_output(self, logits):
        def sparse2dense(tensor, shape):
            tensor = tf.sparse.reset_shape(tensor, shape)
            tensor = tf.sparse.to_dense(tensor, default_value=-1)
            tensor = tf.cast(tensor, tf.float32)
            return tensor

        batch_size = tf.shape(logits)[0]
        max_width = tf.shape(logits)[1]
        logit_length = tf.fill([batch_size], max_width)
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(logits, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_pred = sparse2dense(decoded[0], [batch_size, max_width])

        return y_pred

    def postprocess_remove_blank(self, pred, blank=-1):
        pred = np.array(pred, np.int32)
        results = []
        for p in pred:
            p_no_blank = p[np.where(p != -1)[0]]
            result = ''.join([self.num_to_char[n] for n in p_no_blank])

            results.append(result)
        return results

    def resize_pad(self, img, img_shape):
        h, w, c = img.shape
        height, width, _ = img_shape
        ratio_h, ratio_w = height / h, width / w
        ratio = min(ratio_h, ratio_w)
        new_h, new_w = int(ratio * h), int(ratio * w)
        new_img = cv2.resize(img, (new_w, new_h))

        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[(height - new_h) // 2:(height - new_h) // 2 + new_h,
               (width - new_w) // 2:(width - new_w) // 2 + new_w, :] = new_img
        return canvas

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_pad(img, params.img_shape)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)

        output = self.model(img, training=False)
        output = self.parse_output(output)
        result = self.postprocess_remove_blank(output)

        return result[0]
