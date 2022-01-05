import numpy as np
import cv2
from tensorflow import keras


class Basebag(object):
    def __init__(self, model_path):
        self.class_name = ['No_bag', 'full_bag', 'one_and_three_bag', 'one_and_two_bag', 'one_bag', 'three_bag', 'two_and_three_bag', 'two_bag']
        self.load_model(model_path)

    def load_model(self,model_path):
        self.Bag_model =  keras.models.load_model(model_path)

    def img_process(self, img):
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        img = img / 255.0
        return img

    def predict(self, img , nobag_img):
        test_images = list()
        img = self.img_process(img)
        nobag_img = self.img_process(nobag_img)
        test_images.append(np.concatenate((img,nobag_img),axis=2))
        test_images = np.array(test_images)
        Bag_predictions = self.Bag_model.predict(test_images)

        return self.class_name[np.argmax(Bag_predictions)]
