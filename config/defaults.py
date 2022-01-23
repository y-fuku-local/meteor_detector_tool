from yacs.config import CfgNode as CN

_C = CN()
_C.ADMINISTRATOR_SETTING = CN()
_C.ADMINISTRATOR_SETTING.mask_th = 50
_C.ADMINISTRATOR_SETTING.update_frame = 30
_C.ADMINISTRATOR_SETTING.median_ksize = 5
_C.ADMINISTRATOR_SETTING.det_th = 30
_C.ADMINISTRATOR_SETTING.log_folder = "log"
_C.ADMINISTRATOR_SETTING.bg_folder = "bg"
_C.ADMINISTRATOR_SETTING.meteor_pxl_log = "meteor_pxl.csv"

_C.USER_SETTING = CN()
_C.USER_SETTING.input_video_folder = "D:/git/python_sample/meteor_detector_tool/videos"
_C.USER_SETTING.output_video_folder = "D:/git/python_sample/meteor_detector_tool/output_videos"
_C.USER_SETTING.video_extension = ".mp4"
_C.USER_SETTING.img_range_x = [0, 960]
_C.USER_SETTING.img_range_y = [0, 350]
