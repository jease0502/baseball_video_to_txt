import cv2
import numpy as np
from tensorflow import keras


class Basebag(object):
    def __init__(self, model_path):
        self.class_name = ['No_bag', 'full_bag', 'one_and_three_bag',
                           'one_and_two_bag', 'one_bag', 'three_bag', 'two_and_three_bag', 'two_bag']
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.Bag_model = keras.models.load_model(self.model_path)

    def predict(self, img):
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        test_image = np.array(img)
        test_image = test_image / 255.0
        Bag_predictions = self.Bag_model.predict(
            np.array(test_image.reshape(-1, 64, 64, 3)))
        return self.class_name[np.argmax(Bag_predictions)]
