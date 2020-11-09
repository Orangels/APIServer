import threading
import time
import traceback
import uuid

from modules.models.business_logic.business_params import get_business_params
from modules.utils.business_utils import update_track_info, save_person_info_to_local, save_best_person_info_to_local, \
    get_bgpic_picinfo
from modules.utils.customer_check import CustomerCheck
from modules.utils.http_api import json_data_post


class BusinessLogic:
    def __init__(self, media_id):
        self.media_id = media_id
        self.uuid = uuid.uuid4()

        # 获取业务参数
        self.business_params = {}
        s = threading.Thread(target=get_business_params, args=(self.media_id, self.business_params,))
        s.setDaemon(True)
        s.start()
        # 等待1秒用于获取配置
        time.sleep(1)

        # 类
        self.customer_check = CustomerCheck()

        # 数据
        self.track_infos = {}
        self.in_total = 0
        self.out_total = 0
        self.in_current = 0
        self.out_current = 0
        self.headpic_current = []
        self.bgpic_current = {}
        self.heartbeat_current = 0

    # 删除已经消失或是长时间未出现(错误的冗余数据)的track_id对应的信息，释放空间
    def delete_track_info(self, image_info):

        max_mismatch = self.business_params["TRACK"]["MAX_MISMATCH_TIMES"] * 10
        del_track = []
        for track_id in self.track_infos.keys():
            track_info = self.track_infos[track_id]
            if image_info.get("image_id") - track_info["last_image_id"] > max_mismatch \
                    or track_id in image_info.get("delete_track_ids"):
                del_track.append(track_id)

        for track_id in del_track:
            del self.track_infos[track_id]

    # 将当前图片的信息加入track_infos
    def update_track_infos(self, img, image_info):
        try:
            image_id = image_info.get("image_id")
            for annotation_info in image_info.get("annotations"):
                track_id = annotation_info.get("track_id")
                if track_id in self.track_infos:
                    track_info = self.track_infos[track_id]
                else:
                    track_info = {}
                    track_info["track_info"] = {}

                track_info["last_image_id"] = image_id
                track_info["annotation_info"] = annotation_info

                # 维护track信息
                track_info["track_info"] = update_track_info(img, annotation_info, track_info["track_info"],
                                                             self.business_params)

                self.track_infos[track_id] = track_info

            # 轨迹消失时，将end_flag改成True
            for track_id in image_info.get("delete_track_ids"):
                if track_id in self.track_infos:
                    if self.track_infos[track_id]["track_info"]:
                        self.track_infos[track_id]["track_info"]["end_flag"] = True

        except Exception as e:
            print(traceback.format_exc())

    # 进出店判断，生成需要上传LMA计数、图片的数据
    def get_customer_info(self, vaild_track=False):

        in_current = 0
        out_current = 0
        headpic_current = []

        for track_id, track_info in self.track_infos.items():

            if track_info["track_info"].get("end_flag") and track_info["track_info"].get("track_points"):
                # 判断进店还是出店
                in_tmp, out_tmp = self.customer_check.deter_in_out(track_info["track_info"]['track_points'],
                                                                   self.business_params)
                in_current += in_tmp
                out_current += out_tmp

                # 判断是否有人脸
                if track_info["track_info"]["best_face_info"]:
                    # 形成有效track，vaild_track为True
                    track_info["track_info"]["best_face_info"]["vaild_track"] = bool(
                        in_current or out_current or vaild_track)
                    headpic_current.append(track_info["track_info"]["best_face_info"])
                    track_info["track_info"]["best_face_info"] = {}

                track_info["track_info"]["track_index"] += 1
                track_info["track_info"]["track_points"] = []
                track_info["track_info"]["end_flag"] = False

        self.in_current = in_current
        self.out_current = out_current
        self.in_total += self.in_current
        self.out_total += self.out_current
        self.headpic_current = headpic_current

    # 人脸图片过滤并保存本地
    def save_person_pic(self, img, image_info):
        try:
            if self.business_params["DATA"]["SAVE_LOCAL_PIC"]:
                dirname = self.business_params["DATA"]["LOCAL_PATH"] + (
                    self.business_params["INFO"]["DEVICE_MAC"] if self.business_params["INFO"]["DEVICE_MAC"] else str(
                        self.media_id))
                image_id = image_info.get("image_id")
                for annotation_info in image_info.get("annotations"):
                    track_id = annotation_info.get("track_id")
                    if track_id in self.track_infos:
                        if self.track_infos[track_id]["annotation_info"].get("score") \
                                and self.track_infos[track_id]["annotation_info"].get("score") > 0:
                            extract_info = {
                                "uuid": self.uuid,
                                "image_id": image_id,
                                "media_id": self.media_id,
                                "dirname": dirname,
                                "valid_track": False,
                                "track_index": self.track_infos[track_id]["track_info"]["track_index"]
                            }

                            save_person_info_to_local(img, self.track_infos[track_id], extract_info,
                                                      self.business_params["OTHER"])


        except Exception as e:
            print(traceback.format_exc())

    # 有效track，保存质量最好的图片到本地
    def save_best_person_pic(self, image_info):
        try:
            if self.business_params["DATA"]["SAVE_LOCAL_PIC"]:
                dirname = self.business_params["DATA"]["LOCAL_PATH"] + (
                    self.business_params["INFO"]["DEVICE_MAC"] if self.business_params["INFO"]["DEVICE_MAC"] else str(
                        self.media_id))
                image_id = image_info.get("image_id")

                if self.headpic_current:
                    for best_face_info in self.headpic_current:
                        if best_face_info["person"]["score"] > 0 and best_face_info["vaild_track"]:
                            extract_info = {
                                "uuid": self.uuid,
                                "image_id": image_id,
                                "media_id": self.media_id,
                                "dirname": dirname,
                                "valid_track": True,
                                "track_index": best_face_info["track_index"]
                            }

                            save_best_person_info_to_local(best_face_info, extract_info)

        except Exception as e:
            print(traceback.format_exc())

    # 人脸数据上报LMA
    def send_headpic(self):
        if self.business_params["DATA"]["UPLOAD_HEADPIC"] and self.headpic_current:
            pics_json = []
            for info in self.headpic_current:
                if info["person"]["score"] > 0:
                    person_data = {
                        "trackId": int(info["person"]["track_id"]),
                        "age": int(info["person"]["age"]),
                        "gender": info["person"]["gender"],
                        "score": info["person"]["score"],
                        "angle": info["person"]["angle"],
                        "width": info["person"]["face_box"][2],
                        "height": info["person"]["face_box"][3],
                        "confidence": info["person"]["face_confidence"]
                    }

                    pic_data = {}
                    pic_data["eventTime"] = int(info["event_time"])
                    pic_data["picFile"] = info["picFile"]
                    pic_data["type"] = "in"
                    pic_data["width"] = info["shape"][1]
                    pic_data["height"] = info["shape"][0]
                    pic_data["person"] = person_data

                    pics_json.append(pic_data)

            if pics_json:
                json_send = {"media_id": self.media_id, "pics": pics_json, "format": 'image/jpeg'}
                json_data_post(json_send, self.business_params["DATA"]["HEADPIC_URL"])

    # 计数数据上报LMA
    def send_pvcount(self):
        if self.business_params["DATA"]["UPLOAD_COUNT"] and (self.in_current or self.out_current):
            current_time = int(time.time())
            json_send = {"media_id": self.media_id, "inNum": self.in_current, "outNum": self.out_current,
                         "totalInNum": self.in_total, "totalOutNum": self.out_total, "eventTime": current_time}
            json_data_post(json_send, self.business_params["DATA"]["COUNT_URL"])

    # 底图数据上报LMA
    def send_bgpic(self, img, image_info):
        if self.business_params["DATA"]["UPLOAD_BGPIC"]:

            current_time = time.time()

            # 保存人头最多的底图
            if not self.bgpic_current or self.bgpic_current['best_head_num'] < image_info['head_num']:
                self.bgpic_current['best_img'] = img.copy()
                self.bgpic_current['best_head_num'] = image_info['head_num']
                self.bgpic_current['best_head_time'] = current_time

            # 满足发送间隔，调用LMA接口，并重置bgpic_current
            if not self.bgpic_current.get("last_save_time") or current_time - self.bgpic_current['last_save_time'] >= \
                    self.business_params["DATA"]["BGPIC_INTERVAL"]:
                encode_img, width, height = get_bgpic_picinfo(self.bgpic_current)

                mac = self.business_params["INFO"]["DEVICE_MAC"] if self.business_params["INFO"]["DEVICE_MAC"] else str(
                    self.media_id)

                json_send = {"media_id": self.media_id, "mac": mac, "picFile": encode_img, "height": height,
                             "width": width, "eventTime": int(self.bgpic_current['best_head_time']),
                             "format": 'image/jpeg'}

                json_data_post(json_send, self.business_params["DATA"]["BGPIC_URL"])

                self.bgpic_current['best_img'] = None
                self.bgpic_current['best_head_num'] = -1
                self.bgpic_current['best_head_time'] = 0
                self.bgpic_current['last_save_time'] = current_time
        else:
            # 关闭底图上报，清空bgpic_current
            self.bgpic_current = {}

    # 心跳数据上报LMA
    def send_heartbeat(self):
        if self.business_params["DATA"]["UPLOAD_HEARTBEAT"]:

            current_time = time.time()
            # 满足发送间隔，调用LMA接口，并更新bgpic_current
            if current_time - self.heartbeat_current >= self.business_params["DATA"]["HEARTBEAT_INTERVAL"]:
                json_send = {"media_id": self.media_id, "eventTime": int(self.heartbeat_current)}

                json_data_post(json_send, self.business_params["DATA"]["HEARTBEAT_URL"])

                self.heartbeat_current = current_time
