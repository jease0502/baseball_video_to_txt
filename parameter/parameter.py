


class Parameter(object):
    picture_path = "../old_code/img/"
    savepath = "../old_code/save/"
    
    dir_BSO =savepath + 'BSO/'
    dir_baseBag = savepath + 'baseBag/'
    dir_boardNumber = savepath + 'boardNumber/'
    dir_out_ball = savepath + 'out_ball/'
    dir_out_word = savepath + 'out_word/'
    dir_teamScore = savepath + 'teamScore/'
    dir_BS = savepath + 'BS/'


    bso_model = "../old_code/model/bso.h5"
    bso_number_model = "../old_code/model/bso_number.h5"
    outball_model = "../old_code/model/outballfine.h5"
    Bag_model_path = "../old_code/model/CNNv5"
    team_score_path = "../old_code/model/teamscore.h5"
    bs_model = '../old_code/model/bstf.h5'

    scoreboard_model_path =  '../old_code/model/scoreboard.h5'
    scoreboardanchors_path =  '../old_code/model_data/yolo_anchors.txt'
    scoreboardclasses_path = '../old_code/model_data/scoreboard_classes.txt'
    scoreboard_score = 0.5
    scoreboard_iou = 0.3
    scoreboard_max_boxes = 100
    scoreboard_model_image_size = (608, 608)
    scoreboard_letterbox_image = False

    out_xml = "../old_code/output/tmp.xml"
    out_img = "../old_code/output/tmp.jpg"

    out_txt = "out.txt"

    