from load_modol import Baseball_Video_to_Txt
from cutimage import Cutimage
from parameter import Parameter as parm

import cv2
import xml.dom.minidom as minidom
import os
import numpy as np
import time


class Workflow:
    def __init__(self):
        self.model = Baseball_Video_to_Txt()
        self.model.load_model()
        self.start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.xml_path = parm.out_xml
        self.savePath = parm.savepath
        with open(parm.savepath + "scoreboard" + self.start_time + ".csv", 'a') as f:
            f.write("TeamScore_1,TeamScore_2,B,S,O,Innings,Bases,Velocity\n")
            f.close()
        self.task_dict = {
            'baseBag': parm.dir_baseBag,
            'basebag': parm.dir_baseBag,
            'boardNumber': parm.dir_boardNumber,
            'bso': parm.dir_BSO,
            'out_ball': parm.dir_out_ball,
            'out_word': parm.dir_out_word,
            'B-S': parm.dir_BS,
            'teamScore': parm.dir_teamScore,
            'velocity': parm.dir_velocity
        }

        self.scoreboard_data = {
            "TeamScore_1": " ",
            "TeamScore_2": " ",
            "B": " ",
            "S": " ",
            "O": " ",
            "Innings": " ",
            "Bases": " ",
            "Velocity": " ",
        }

    def run(self, img):
        self.count = 0
        self.last_name = ""
        self.img = img
        scoreboard_img = self.model.detect_scoreboard(self.img)
        self.model.detect_cus(scoreboard_img)
        self.xml_anatomy(scoreboard_img)
        # self.count = 0
        # self.last_name = ""
        # self.model.detect_team_velocity(scoreboard_img)
        # self.xml_anatomy(scoreboard_img)
        # self.data_save_to_csv()

        print(self.scoreboard_data)

    def xml_anatomy(self, img):
        xmlBase = os.path.basename(self.xml_path)
        xmlName = os.path.splitext(xmlBase)[0]
        doc = minidom.parse(self.xml_path)
        root = doc.documentElement
        size = root.getElementsByTagName('object')

        for i in range(size.length):
            name = size[i].getElementsByTagName(
                'name')[0].childNodes[0].nodeValue
            xmin = size[i].getElementsByTagName(
                'xmin')[0].childNodes[0].nodeValue
            ymin = size[i].getElementsByTagName(
                'ymin')[0].childNodes[0].nodeValue
            xmax = size[i].getElementsByTagName(
                'xmax')[0].childNodes[0].nodeValue
            ymax = size[i].getElementsByTagName(
                'ymax')[0].childNodes[0].nodeValue

            if name in self.task_dict.keys():
                cutImage = img[int(ymin): int(ymax), int(xmin): int(xmax)]
                self.count = self.count + 1 if self.last_name == name else self.count
                data = self.model.position_to_txt(name, cutImage)
                self.data_conversion(name, data, self.count)
            self.last_name = name

    def data_conversion(self, name, data, count):
        if(name == "baseBag" or name == "basebag"):
            self.scoreboard_data["Bases"] = data
        elif(name == "boardNumber"):
            self.scoreboard_data["Innings"] = data
        elif(name == "bso"):
            if(data[0] == "B"):
                self.scoreboard_data["B"] = str(data[1])
            elif(data[0] == "S"):
                self.scoreboard_data["S"] = str(data[1])
            elif(data[0] == "O"):
                self.scoreboard_data["O"] = str(data[1])
        elif(name == "out_ball"):
            self.scoreboard_data["O"] = str(data)
        elif(name == "out_word"):
            self.scoreboard_data["O"] = str(data)
        elif(name == "B-S"):
            self.scoreboard_data["B"] = str(data[0])
            self.scoreboard_data["S"] = str(data[2])
        elif(name == "teamScore"):
            if count == 0:
                self.scoreboard_data["TeamScore_1"] = data[0]
            else:
                self.scoreboard_data["TeamScore_2"] = data[0]
        elif(name == "velocity"):
            self.scoreboard_data["Velocity"] = data

    def data_save_to_csv(self):
        """
        save data to csv
        """
        with open(parm.savepath + "scoreboard" + self.start_time + ".csv", 'a') as f:
            f.write(self.scoreboard_data["TeamScore_1"]+",")
            f.write(self.scoreboard_data["TeamScore_2"]+",")
            f.write(self.scoreboard_data["B"]+",")
            f.write(self.scoreboard_data["S"]+",")
            f.write(self.scoreboard_data["O"]+",")
            f.write(self.scoreboard_data["Innings"]+",")
            f.write(self.scoreboard_data["Bases"]+",")
            f.write(self.scoreboard_data["Velocity"]+"\n")
            f.close()
