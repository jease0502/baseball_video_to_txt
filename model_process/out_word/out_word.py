import tensorflow as tf
import tensorflow.keras as K
from cv2 import cv2




class Out_word(object):
    def __init__(self,model_path):
        self.shape = (50,80)
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model = K.models.load_model(self.model_path)

    def predict(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,self.shape[::-1])

        img = tf.convert_to_tensor(img,tf.float32)
        img /= 255.
        img = tf.expand_dims(img,axis = 0)

        result = self.model(img,training = False)
        num = tf.argmax(result,axis = 1)
        num = num.numpy()[0]
        return num