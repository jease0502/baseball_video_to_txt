from model_process.yolo.scoreboardyolo import YOLO as scoreboard_yolo
from model_process.yolo.cus_yolo import YOLO as cus_yolo
from model_process.yolo.predict_xml import Predict_xml
from model_process.basebag.basebag import Basebag
from model_process.bs.bs import Bs
from model_process.bso.bso import Bso
from model_process.out_ball.out_ball import Out_ball
from model_process.out_word.out_word import Out_word
from model_process.teamscore.teamscore import Team_score
from model_process.velocity.velocity import Velocity
from model_process.boardnumber.boardnumber import BoardNumber
from model_process.yolo.teamyolo import YOLO as team_yolo

from parameter import Parameter as parm

import cv2


class Baseball_Video_to_Txt(object):

    def load_model(self):
        self.scoreboard_yolo = scoreboard_yolo()
        self.cus_yolo = cus_yolo()
        self.team_vel_yolo = team_yolo()
        self.basebag = Basebag("")
        self.bs = Bs(
            "/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/bs.h5")
        self.bso = Bso("/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/bso.h5",
                       "/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/bso_number.h5")
        self.out_ball = Out_ball(
            "/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/outballfine.h5")
        self.out_word = Out_word(
            "/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/out_word.h5")
        self.team_score = Team_score(
            "/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/teamscore.h5")
        self.boardNumber = BoardNumber("/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/crnnbn.h5",
                                       "/home/d0752870/baseball_video_to_txt/baseball_video_to_txt/model/bn.h5")
        # self.velocity = Velocity("/home/d0752870/baseball_video_to_txt/jease0502-baseball_video_to_txt/model/velocity.h5")
        print("load done")

    def position_to_txt(self, position, img):
        self.task_call_dict = {
            'baseBag': self.detect_basebag,
            'basebag': self.detect_basebag,
            'boardNumber': self.detect_board_number,
            'bso': self.detect_bso,
            'out_ball': self.detect_out_ball,
            'out_word': self.detect_out_word,
            'B-S': self.detect_bs,
            'teamScore': self.detect_team_score
        }
        data = self.task_call_dict[position](img)
        return data

    def detect_scoreboard(self, img):
        return self.scoreboard_yolo.detect_image(img)

    def detect_cus(self, img):
        position = Predict_xml(self.cus_yolo, parm.out_img, parm.out_xml, True)
        position.predict(img)

    def detect_team_velocity(self, img):
        position = Predict_xml(
            self.team_vel_yolo, parm.out_img, parm.out_xml, True)
        position.predict(img)

    def detect_basebag(self, img):
        return self.basebag.predict(img)

    def detect_bs(self, img):
        return self.bs.predict(img)

    def detect_bso(self, img):
        return self.bso.predict(img)

    def detect_out_ball(self, img):
        return self.out_ball.predict(img)

    def detect_out_word(self, img):
        return self.out_word.predict(img)

    def detect_team_score(self, img):
        return self.team_score.predict(img)

    def detect_velocity(self, img):
        return self.velocity.predict(img)

    def detect_board_number(self, img):
        return self.boardNumber.predict(img)
