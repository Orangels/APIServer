import glob
import os
import logging
import time
import traceback

import cv2
import numpy as np

from modules.inferface.frame_draw import FrameDraw
from modules.inferface.model_deal import ModelDeal
from modules.inferface.business_logic import BusinessLogic
from modules.utils.decorators import log_process_time

LOG_DIR = "/srv/Data/projectLogs/"
DEBUG_LEVEL = logging.INFO


class ObjectApi():
    def __init__(self, camId=0):
        self.camId = camId
        if not os.path.isdir(LOG_DIR):
            os.makedirs(LOG_DIR)

        self.handler = logging.FileHandler('{}cam_{}.log'.format(LOG_DIR, str(camId)), mode="a", encoding='utf-8')
        self.logger = logging.getLogger(str(camId))
        self.logger.setLevel(DEBUG_LEVEL)
        self.logger.addHandler(self.handler)

        self.model_deal = ModelDeal()
        self.business_logic = BusinessLogic(self.camId)
        self.frame_draw = FrameDraw()
        self.image_id = 0

        self.logger.info("Init {} -- {}".format(self.__class__.__name__, self.camId))

        self.now_time = time.time()

    @log_process_time("get_result")
    def get_result(self, img, bbox, trackIDs, deleteIDs, kpts, ageGender):

        self.logger.info("{} inferface".format(format(time.time() - self.now_time, '.4f')))
        self.now_time = time.time()

        self.image_id += 1
        start = time.time()
        try:
            # 非工作时间，不进行业务处理，直接返回img
            if not self.business_logic.business_params["OTHER"]["WORK_FLAG"]:
                return img[0]

            frame = img[0]
            bbox_np = bbox[0]
            trackIDs_np = trackIDs[0]
            deleteIDs_np = deleteIDs[0]
            kpts_np = kpts[0]
            ageGender_np = ageGender[0]

            self.logger.info("{} data_copy".format(format(time.time() - start, '.4f')))
            s = time.time()

            ## 模型后处理
            head_bbox, face_bbox = self.model_deal.face_detection(bbox_np)
            self.logger.info("{} face_detection".format(format(time.time() - s, '.4f')))
            s = time.time()
            track_ids, delete_track_ids = self.model_deal.face_tracker(trackIDs_np, deleteIDs_np)
            self.logger.info("{} face_tracker".format(format(time.time() - s, '.4f')))
            s = time.time()
            ages, genders, norms, features, rescores = self.model_deal.face_age_gender(ageGender_np)
            self.logger.info("{} face_age_gender".format(format(time.time() - s, '.4f')))
            s = time.time()
            kpts_list, pss, angles = self.model_deal.face_key_point(kpts_np)
            self.logger.info("{} face_key_point".format(format(time.time() - s, '.4f')))
            s = time.time()
            image_info = self.model_deal.get_image_info(self.image_id, face_bbox, head_bbox, track_ids,
                                                        delete_track_ids, ages,
                                                        genders, norms, kpts_list, pss, angles)
            self.logger.info("{} get_image_info".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 更新track_infos
            self.business_logic.update_track_infos(frame, image_info)
            self.logger.info("{} update_track_infos".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 保存图片到本地
            self.business_logic.save_person_pic(frame, image_info)
            self.logger.info("{} save_person_pic".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 底图数据上报LMA
            self.business_logic.send_bgpic(frame, image_info)
            self.logger.info("{} send_bgpic".format(format(time.time() - s, '.4f')))
            s = time.time()

            ## 框线的绘制通过business_params["DRAW"]来决定

            # 绘制人脸框、人头框、轨迹
            frame = self.frame_draw.draw_box_track(frame, track_ids, self.business_logic.track_infos,
                                                   self.business_logic.business_params["DRAW"])
            self.logger.info("{} draw_box_track".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 绘制人脸框、人头框、3D关键点
            frame = self.frame_draw.draw_box_key_point(frame, image_info, self.business_logic.business_params["DRAW"])
            self.logger.info("{} draw_box_key_point".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 绘制性别年龄
            frame = self.frame_draw.draw_age_gender(frame, image_info, self.business_logic.business_params["DRAW"])
            self.logger.info("{} draw_age_gender".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 进出店判断，生成需要上传LMA计数、图片的数据
            self.business_logic.get_customer_info()
            self.logger.info("{} get_customer_info".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 计数数据上报LMA
            self.business_logic.send_pvcount()
            self.logger.info("{} send_pvcount".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 人脸数据上报LMA
            self.business_logic.send_headpic()
            self.logger.info("{} send_headpic".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 心跳数据上报LMA
            self.business_logic.send_heartbeat()
            self.logger.info("{} send_heartbeat".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 有效track，保存质量最好的图片到本地
            self.business_logic.save_best_person_pic(image_info)
            self.logger.info("{} save_best_person_pic".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 绘制进出店人次
            frame = self.frame_draw.draw_customer_num(frame, self.business_logic.in_total,
                                                      self.business_logic.out_total,
                                                      self.business_logic.business_params["DRAW"])
            self.logger.info("{} draw_customer_num".format(format(time.time() - s, '.4f')))
            s = time.time()

            # 删除track_infos中的无效数据
            self.business_logic.delete_track_info(image_info)
            self.logger.info("{} delete_track_info".format(format(time.time() - s, '.4f')))

            self.logger.info("{} get_result".format(format(time.time() - start, '.4f')))

            return frame
        except Exception as e:
            self.logger.exception(traceback.format_exc())
            return img[0]

    def __del__(self):
        print(self.__class__.__name__)


def single_test(cam_id, dirname):
    api = ObjectApi(cam_id)

    img = cv2.imread(dirname + "img.jpg"),
    bbox = np.load(dirname + "bbox.npy"),
    trackIDs = np.load(dirname + "trackIDs.npy"),
    kpts = np.load(dirname + "kpts.npy"),
    ageGender = np.load(dirname + "ageGender.npy"),
    deleteIDs = np.load(dirname + "deleteIDs.npy"),

    api.get_result(img, bbox, trackIDs, deleteIDs, kpts, ageGender)
    api.get_result(img, bbox, trackIDs, deleteIDs, kpts, ageGender)
    api.get_result(img, bbox, trackIDs, deleteIDs, kpts, ageGender)


def multi_test(cam_id, dirname):
    api = ObjectApi(cam_id)

    for img_file in glob.glob(dirname + "*.jpg"):
        img = cv2.imread(img_file),
        bbox = np.load(img_file.replace("img", "bbox").replace(".jpg", ".npy")),
        trackIDs = np.load(img_file.replace("img", "trackIDs").replace(".jpg", ".npy")),
        kpts = np.load(img_file.replace("img", "kpts").replace(".jpg", ".npy")),
        ageGender = np.load(img_file.replace("img", "ageGender").replace(".jpg", ".npy")),
        deleteIDs = np.load(img_file.replace("img", "deleteIDs").replace(".jpg", ".npy")),

        api.get_result(img, bbox, trackIDs, deleteIDs, kpts, ageGender)


if __name__ == '__main__':
    # single_test(0, "npy/")
    multi_test(0, "npy_output/")
