class Parameter(object):

#     picture_path =  "../yolov4-tf2/VOCdevkit/VOC2007/JPEGImages/"
    picture_path = "img/"
    # savepath = "/nfs/Projects/baseball/scoreboard/cut0906/"
#     savepath = "/home/d0752870/jease/baseball_video_to_txt/output/"
    savepath = "save/"
    
    dir_BSO =savepath + 'BSO/'
    dir_baseBag = savepath + 'baseBag/'
    dir_boardNumber = savepath + 'boardNumber/'
    dir_out_ball = savepath + 'out_ball/'
    dir_out_word = savepath + 'out_word/'
    dir_teamScore = savepath + 'teamScore/'
    dir_BS = savepath + 'BS/'
    dir_velocity = savepath + 'velocity/'


    bso_model = "model/bso.h5"
    bso_number_model = "model/bso_number.h5"
    outball_model = "model/outballfine.h5"
    Bag_model_path = "model/CNNv5"
    team_score_path = "model/teamscore.h5"
    bs_model = 'model/bstf.h5'

    scoreboard_model_path =  'model/scoreboard.h5'
    scoreboardanchors_path =  'model_data/yolo_anchors.txt'
    scoreboardclasses_path = 'model_data/scoreboard_classes.txt'
    scoreboard_score = 0.5
    scoreboard_iou = 0.3
    scoreboard_max_boxes = 100
    scoreboard_model_image_size = (608, 608)
    scoreboard_letterbox_image = False

    out_xml = "output/tmp.xml"
    out_img = "output/tmp.jpg"

    out_txt = "out.txt"

    