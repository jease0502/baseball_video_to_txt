import xml.dom.minidom as minidom


class Cutimage(object):

    def __init__(self , xml_path, savePath):
        self.xml_path = xml_path
        self.savePath = savePath   

    def xml(self):
        doc = minidom.parse(self.xml_path)
        root = doc.documentElement
        size = root.getElementsByTagName('object')
        
            
        for i in range(size.length):
            name = size[i].getElementsByTagName('name')[0].childNodes[0].nodeValue
            xmin = size[i].getElementsByTagName('xmin')[0].childNodes[0].nodeValue
            ymin = size[i].getElementsByTagName('ymin')[0].childNodes[0].nodeValue
            xmax = size[i].getElementsByTagName('xmax')[0].childNodes[0].nodeValue
            ymax = size[i].getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        
            if name in self.task_dict.keys():
                cutImage = self.img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
                count = count + 1 if last_name == name else count
                cv2.imwrite(self.task_dict[name] + xmlName + str(count)  + '.jpg', cutImage)
                # print(np.shape(cutImage))
                data = self.Predict2Txt.predict_txt(name,cutImage)
                print(data)
                f.write(name + " ： " + str(data) + "\n")
        
        return size
