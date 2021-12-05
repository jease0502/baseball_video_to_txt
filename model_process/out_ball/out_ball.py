import tensorflow as tf
import cv2

class Out_ball(object):
    def __init__(self,model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)


    def predict(self, img):
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        img = tf.cast(img,tf.float32) / 255.0
        img = tf.expand_dims(img,axis = 0)
        y_pred = self.model.predict_classes(img)
        return y_pred[0]