import copy
import sys
import time
import traceback

import mongoengine

LMA_PATH = "/srv/LocalManagementApp"
sys.path.append(LMA_PATH)

from conf.config import default_business_params, draw_dict
from model import MediaInfo, CommonConfig
from modules.utils.customer_check import CustomerCheck


def get_work_flag():
    work_flag = True
    try:
        cc = CommonConfig.objects.get()
        worktime_config = cc.worktime_config
        start_worktime = worktime_config.start_worktime
        end_worktime = worktime_config.end_worktime
        if (end_worktime - start_worktime) >= 86400:
            work_flag = True
        else:
            now = int(time.time() - time.mktime(time.strptime(time.strftime('%Y-%m-%d 00:00:00'), '%Y-%m-%d %H:%M:%S')))
            if end_worktime >= 86400:
                work_flag = True if (now >= start_worktime) or (now <= end_worktime % 86400) else False
            else:
                work_flag = True if (now >= start_worktime) and (now <= end_worktime) else False
    except Exception as e:
        print(traceback.format_exc())
    return work_flag


def update_params(media_id, business_params, customer_check):
    try:
        # media_info
        media = MediaInfo.objects.get(media_id=media_id)
        if media:
            # INFO
            business_params["INFO"]["DEVICE_MAC"] = media.media_mac

            # ASSESS
            assess_params = {
                "MIN_SIZE": media.buss_params.buss_general_params.buss__assess__min_size,
                "MID_SIZE": business_params["ASSESS"]["MID_SIZE"],
                "MAX_SIZE": media.buss_params.buss_advance_params.buss__assess__max_size,
                "MIN_BOX_SCORE": media.buss_params.buss_general_params.buss__assess__min_box_score,
                "MAX_ANGLE_YAW": media.buss_params.buss_general_params.buss__assess__max_angle_yaw,
                "MAX_ANGLE_PITCH": media.buss_params.buss_general_params.buss__assess__max_angle_pitch,
                "MAX_ANGLE_ROLL": media.buss_params.buss_advance_params.buss__assess__max_angle_roll,
                "MIN_BRIGHTNESS": media.buss_params.buss_advance_params.buss__assess__min_brightness,
                "MAX_BRIGHTNESS": media.buss_params.buss_advance_params.buss__assess__max_brightness,
                "L2_NORM": business_params["ASSESS"]["L2_NORM"]
            }
            business_params.update({"ASSESS": assess_params})

            # COUNT

            # 没画识别区域，默认为全屏的5%~95%，四周留5%的空白
            if media.frame_params.roi_area:
                roi_area = customer_check.find_rect(
                    customer_check.get_cordinate(media.frame_params.roi_area, business_params["IMAGE"]["SHAPE"]),
                    business_params["IMAGE"]["SHAPE"])
            else:
                width_blank = business_params["IMAGE"]["SHAPE"][1] // 20
                height_blank = business_params["IMAGE"]["SHAPE"][0] // 20

                roi_area = ([width_blank, height_blank], [business_params["IMAGE"]["SHAPE"][1] - width_blank,
                                                          business_params["IMAGE"]["SHAPE"][0] - height_blank])

            # 没画进门方向，默认为识别区域的x中心点，y的1/4到3/4
            if media.frame_params.entrance_direction:
                direction = customer_check.get_cordinate(media.frame_params.entrance_direction,
                                                         business_params["IMAGE"]["SHAPE"])
            else:
                direction = customer_check.find_direct(roi_area)

            # 没画进门线，默认为全屏的中间
            if media.frame_params.entrance_line:
                entrance_line = customer_check.get_cordinate(media.frame_params.entrance_line,
                                                             business_params["IMAGE"]["SHAPE"])
            else:
                entrance_line = ([0, business_params["IMAGE"]["SHAPE"][0] // 2],
                                 [business_params["IMAGE"]["SHAPE"][1], business_params["IMAGE"]["SHAPE"][0] // 2])

            count_params = {
                "ANGLE": business_params["COUNT"]["ANGLE"],
                "ENTRANCE_DIRECTION": direction,
                "ENTRANCE_LINE": entrance_line,
                "ROI_AREA": roi_area,
                "VECTOR_LEN": business_params["COUNT"]["VECTOR_LEN"]
            }

            if count_params != business_params["COUNT"]:
                business_params.update({"COUNT": count_params})

                # 没画进门线，或是进门线和识别区域没有形成有效进出区域时，默认为符合条件
                if business_params["COUNT"]["ENTRANCE_LINE"]:
                    out_area, in_area = customer_check.find_inout_area(business_params["COUNT"]["ROI_AREA"],
                                                                       business_params["COUNT"]["ENTRANCE_DIRECTION"],
                                                                       business_params["COUNT"]["ENTRANCE_LINE"])
                else:
                    out_area, in_area = None, None

                # DEAL
                deal_params = {
                    "IN_AREA": in_area,
                    "OUT_AREA": out_area
                }

                business_params.update({"DEAL": deal_params})

            business_params["TRACK"][
                "MAX_MISMATCH_TIMES"] = media.buss_params.buss_advance_params.buss__track__max_mismatch_times

            # 回显模式改变，更新Draw
            display_mode = media.buss_params.buss_general_params.buss__other__display_mode
            if display_mode != business_params["OTHER"]["DISPLAY_MODE"]:
                for key, value in draw_dict[display_mode].items():
                    business_params["DRAW"][key] = value

                business_params["OTHER"][
                    "DISPLAY_MODE"] = media.buss_params.buss_general_params.buss__other__display_mode

            business_params["OTHER"]["KPS_ON"] = media.buss_params.buss_advance_params.buss__other__kps_on

            business_params["OTHER"]["WORK_FLAG"] = get_work_flag()

            business_params["DATA"]["UPLOAD_BGPIC"] = media.buss_params.buss_other_params.buss__other__save_best_img
            business_params["DATA"][
                "BGPIC_INTERVAL"] = media.buss_params.buss_other_params.buss__other__save_best_img_time_gap

    except Exception as e:
        print(traceback.format_exc())


def get_business_params(media_id, business_params):
    business_params.update(copy.deepcopy(default_business_params))
    try:
        db_config = business_params["MONGO"]
        mongoengine.connect(db=db_config["DB_NAME"], host='{}:{}'.format(db_config["DB_HOST"], db_config["DB_PORT"]))

        customer_check = CustomerCheck()

        while True:
            update_params(media_id, business_params, customer_check)
            time.sleep(10)

    except Exception as e:
        print(traceback.format_exc())


if __name__ == "__main__":
    business_params = {}
    media_id = 0
    get_business_params(media_id, business_params)
    print(business_params)
