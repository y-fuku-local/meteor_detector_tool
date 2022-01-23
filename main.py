import cv2
import os
import glob
import argparse
import statistics
import shutil
import logging
import sys

from config import cfg

class MeteorDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_video_info(self, file_name):
        cap = cv2.VideoCapture(file_name)
        self.totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def get_bg_image(self, file_name):
        cap = cv2.VideoCapture(file_name)
        ret, bg = cap.read()
        gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        cap.release()
        if True:
            out_folder_name = cfg.ADMINISTRATOR_SETTING.log_folder + '/' + cfg.ADMINISTRATOR_SETTING.bg_folder
            out_img_name = out_folder_name + '/bg_' + os.path.basename(file_name).split(self.cfg.USER_SETTING.video_extension)[0] + '.png'
            if not os.path.isdir(out_folder_name):
                os.makedirs(out_folder_name)
            cv2.imwrite(out_img_name, gray_bg)
        return gray_bg

    def get_bbox(self, gray_image, bg):
        crop_gray_image = gray_image[self.cfg.USER_SETTING.img_range_y[0]:self.cfg.USER_SETTING.img_range_y[1], \
                                     self.cfg.USER_SETTING.img_range_x[0]:self.cfg.USER_SETTING.img_range_x[1]]
        crop_bg         = bg[self.cfg.USER_SETTING.img_range_y[0]:self.cfg.USER_SETTING.img_range_y[1], \
                             self.cfg.USER_SETTING.img_range_x[0]:self.cfg.USER_SETTING.img_range_x[1]]

        mask = cv2.absdiff(crop_gray_image, crop_bg)

        mask[mask <  self.cfg.ADMINISTRATOR_SETTING.mask_th] = 0
        mask[mask >= self.cfg.ADMINISTRATOR_SETTING.mask_th] = 255
        mask = cv2.medianBlur(mask, ksize = self.cfg.ADMINISTRATOR_SETTING.median_ksize)
        labels, label_imgs, bbox, center = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        return labels, label_imgs, bbox, center

    def save_videos(self, file_name, bg):
        cap = cv2.VideoCapture(file_name)
        frame_counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                break
            
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            labels, label_imgs, bbox, center = self.get_bbox(gray_image, bg)

            if labels > 1:
                if max(bbox[1:bbox.shape[0],4]) >= self.cfg.ADMINISTRATOR_SETTING.det_th:
                    out_video_name = self.cfg.USER_SETTING.output_video_folder + '/' + \
                                     os.path.basename(file_name).split(self.cfg.USER_SETTING.video_extension)[0] + \
                                     '_' + str(frame_counter).zfill(6) + self.cfg.USER_SETTING.video_extension

                    video = cv2.VideoWriter(out_video_name, self.fourcc, self.fps, (self.w, self.h))
                    video.write(gray_image)
                    video_frame_counter = 1
                    pxl_list = list([max(bbox[1:bbox.shape[0],4])])
                    while(1):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_counter += 1
                        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        labels, label_imgs, bbox, center = self.get_bbox(gray_image, bg)
                        if labels <= 1 or max(bbox[1:bbox.shape[0],4]) < self.cfg.ADMINISTRATOR_SETTING.det_th:
                            break
                        else:
                            video.write(gray_image)
                            video_frame_counter += 1
                            pxl_list.append(max(bbox[1:bbox.shape[0],4]))

                    pxl_mean = statistics.mean(pxl_list)
                    f = open(self.cfg.ADMINISTRATOR_SETTING.log_folder + '/' + self.cfg.ADMINISTRATOR_SETTING.meteor_pxl_log, 'a')
                    str_out = 'file,' + os.path.basename(file_name).split(self.cfg.USER_SETTING.video_extension)[0] + \
                              '_' + str(frame_counter).zfill(6) + ',pxl_mean,' + str(pxl_mean) + ',video_frames,' + str(video_frame_counter)
                    f.write(str_out + "\n")
                    print(str_out)
        cap.release()

def main(cfg):
    if not os.path.isdir(cfg.USER_SETTING.output_video_folder):
        os.makedirs(cfg.USER_SETTING.output_video_folder)
    file_list = glob.glob(cfg.USER_SETTING.input_video_folder + '/*' + cfg.USER_SETTING.video_extension)
    meteor_detector = MeteorDetector(cfg)

    for file_idx in range(len(file_list)):
        file_name = file_list[file_idx]
        meteor_detector.get_video_info(file_name)
        bg = meteor_detector.get_bg_image(file_name)
        meteor_detector.save_videos(file_name, bg)

def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger

def get_cfg():
    parser = argparse.ArgumentParser(
        description="parser test"
    )
    parser.add_argument(
        "--cfg",
        default="../param.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    if os.path.isdir(cfg.ADMINISTRATOR_SETTING.log_folder):
        shutil.rmtree(cfg.ADMINISTRATOR_SETTING.log_folder)
    os.makedirs(cfg.ADMINISTRATOR_SETTING.log_folder)
    
    with open(os.path.join(cfg.ADMINISTRATOR_SETTING.log_folder + '/config.yaml'), 'w') as f:
        f.write("{}".format(cfg))
    return cfg
if __name__=='__main__':
    main(get_cfg())