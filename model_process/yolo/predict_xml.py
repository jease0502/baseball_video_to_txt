import tensorflow as tf
from PIL import Image
import xml.dom.minidom as minidom
import os
import numpy as np


class Predict_xml(object):
    def __init__(self, yolo, output_img_filename=None, output_xml_filename=None, save_originals=False):
        self.output_img_filename = output_img_filename
        self.output_xml_filename = output_xml_filename
        self.save_originals = save_originals

        if not os.path.exists(os.path.dirname(self.output_img_filename)):
            os.makedirs(os.path.dirname(self.output_img_filename))
        if not os.path.exists(os.path.dirname(self.output_xml_filename)):
            os.makedirs(os.path.dirname(self.output_xml_filename))
        self.yolo = yolo

    def predict(self, img):
        if type(img) == np.ndarray:
            tmp_img = Image.fromarray(img).copy()
        else:
            tmp_img = img.copy()

        org_img = tmp_img.copy()
        w, h = tmp_img.size

        r_image, bbox, _ = self.yolo.detect_image(tmp_img, False)

        if self.output_img_filename is not None and self.output_xml_filename is not None:
            r_image.save(self.output_img_filename)
            if self.save_originals:
                org_img.save(
                    self.output_img_filename[:-4] + '_o' + self.output_img_filename[-4:])
            self.save_xml(os.path.basename(self.output_img_filename),
                          self.output_xml_filename, (h, w, 3), bbox)

    def save_xml(self, filename, output_file_name, img_shape, label):

        h, w, c = img_shape

        doc = minidom.Document()
        annotation = doc.createElement('annotation')

        filename_node = doc.createElement('filename')
        filename_node.appendChild(
            doc.createTextNode(os.path.basename(filename)))
        annotation.appendChild(filename_node)

        size = doc.createElement('size')
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(w)))
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(h)))
        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode(str(c)))

        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)

        annotation.appendChild(size)

        for x_min, y_min, x_max, y_max, obj_name in label:
            x_min = x_min if x_min >= 0 else 0
            y_min = y_min if y_min >= 0 else 0
            x_max = x_max if x_max < w else w
            y_max = y_max if y_max < h else h

            objects = doc.createElement('object')
            object_name = doc.createElement('name')
            object_name.appendChild(doc.createTextNode(obj_name))
            bndbox = doc.createElement('bndbox')
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(str(int(round(x_min)))))
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(str(int(round(y_min)))))
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str(int(round(x_max)))))
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str(int(round(y_max)))))
            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)

            objects.appendChild(object_name)
            objects.appendChild(bndbox)

            annotation.appendChild(objects)

        doc.appendChild(annotation)

        with open(output_file_name, 'w') as f:
            doc.writexml(f, indent='\t', addindent='\t',
                         newl='\n', encoding="utf-8")
