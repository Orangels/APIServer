import base64
import json
import os
import time
import traceback

import cv2
import numpy as np


# 根据添加过滤图片，并返回图片质量分
def get_quantity(face_confidence, angle, norm, face_box, shading, business_params_assess):
    box_size_weight = 2
    angle_weight = 5
    norm_weight = 4
    face_confidence_weight = 5
    MIN_SIZE = business_params_assess["MIN_SIZE"]
    MID_SIZE = business_params_assess["MID_SIZE"]
    MAX_SIZE = business_params_assess["MAX_SIZE"]
    min_side = min(face_box[2], face_box[3])
    if norm:
        norm_score = max(min(norm[0], 1), 0)
    else:
        norm = [1]
        norm_weight = 0
        norm_score = 0

    if angle:
        angle_score = 1 - (abs(angle[0]) / 90 * 0.6 + abs(angle[1]) / 90 * 0.4)
    else:
        angle = [0, 0, 0]
        angle_score = 0
        angle_weight = 0

    if min_side < MIN_SIZE or face_confidence < business_params_assess["MIN_BOX_SCORE"] or abs(angle[0]) > \
            business_params_assess["MAX_ANGLE_YAW"] or \
            abs(angle[1]) > business_params_assess["MAX_ANGLE_PITCH"] or abs(angle[2]) > \
            business_params_assess["MAX_ANGLE_ROLL"] \
            or abs(norm[0]) < business_params_assess["L2_NORM"] or shading < business_params_assess["MIN_BRIGHTNESS"]:
        return 0
    total_weight = box_size_weight + angle_weight + norm_weight + face_confidence_weight
    box_size_score = min((min_side - MIN_SIZE) / (MAX_SIZE - MIN_SIZE), 1)
    if min_side < MID_SIZE:
        return box_size_score * box_size_weight / total_weight
    total_score = (box_size_score * box_size_weight + angle_score * angle_weight +
                   norm_score * norm_weight + face_confidence * face_confidence_weight) / total_weight
    # total_score = total_score * 0.6 + 0.4
    total_score = min(max(total_score, 0), 1)
    return total_score


## 计算光照强度
def cal_brightness(img, face_box):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = img_gray[int(face_box[1]):int(face_box[1] + face_box[3]), int(face_box[0]):int(face_box[0] + face_box[2])]
    return cv2.mean(roi)[0]


# 判断当前人头框的中心点是否在识别区域内，是的话，加入轨迹坐标列表，in_area为True
# 如果有人脸框，计算光照强度和图片质量，维护最好的图片best_face_info
# track保留起始点+最后DRAW_TRACK_NUM-1个点
def update_track_info(img, person, track_info, business_params):
    roi_area = business_params["COUNT"]["ROI_AREA"]

    rect_roi_area = np.array(
        [[roi_area[0][0], roi_area[0][1]], [roi_area[1][0], roi_area[0][1]], [roi_area[1][0], roi_area[1][1]],
         [roi_area[0][0], roi_area[1][1]]])

    business_params_assess = business_params["ASSESS"]
    business_params_other = business_params["OTHER"]
    draw_track_num = business_params["DRAW"]["DRAW_TRACK_NUM"]

    box_xywh = person["head_box"]

    # 人头框中心点(列表)
    position = [int(box_xywh[0] + box_xywh[2] / 2), int(box_xywh[1] + box_xywh[3] / 2)]

    # 在识别区域内，更新update_track_info
    if is_point_in_area(tuple(position), rect_roi_area):
        if track_info:
            track_points = track_info["track_points"]
            track_points.append(position)
            if len(track_points) > draw_track_num + 1:
                track_points = [track_points[0]] + track_points[-draw_track_num:]
            track_info["track_points"] = track_points
            track_info["in_area"] = True
        else:
            track_info = {
                "track_index": 0,
                "track_points": [position],
                "in_area": True,
                "best_face_info": {},
                "end_flag": False
            }

        # 有人脸框，维护best_face_info
        if person["face_box"]:

            # 光照强度
            person["shading"] = cal_brightness(img, person["face_box"])

            # 图片质量
            person["score"] = get_quantity(person["face_confidence"], person["angle"], person["norm"],
                                           person["face_box"], person["shading"], business_params_assess)

            prev_score = track_info.get("best_face_info")["person"]["score"] if track_info.get("best_face_info") else 0

            # 当前图片质量分高，替换
            if person["score"] > prev_score:

                # 半身图
                face_pad_img, offset = get_pad_img(person["head_box"], img)
                if business_params_other["USE_BRIGHTNESS_ENHANCEMENT"]:
                    face_pad_img = birght_img(face_pad_img, business_params_other["BRIGHTNESS_GAIN"])
                pad_img = b64encode(face_pad_img)

                best_face_info = {
                    "event_time": time.time(),
                    "picFile": pad_img,
                    "person": person,
                    "shape": face_pad_img.shape,
                    "offset": offset,
                    "track_index": track_info["track_index"]
                }
                track_info["best_face_info"] = best_face_info

    # 出框，end_flag为True
    elif track_info.get("track_points"):
        track_info["in_area"] = False
        track_info["end_flag"] = True

    return track_info


## 根据人头框向外扩展成半身图
def get_pad_img(rect, img, ratio=(0.15, 0.45, 0.3, 0.3)):
    x, y, w, h = rect
    # pad box to square
    if w > h:
        y = y - (w - h) / 2
    else:
        x = x - (h - w) / 2
        w = h
    img_h, img_w, c = img.shape
    new_x1_ = int(x - w * ratio[2])
    new_y1_ = int(y - w * ratio[0])
    new_x2_ = int(x + w * (1 + ratio[3]))
    new_y2_ = int(y + w * (1 + ratio[1]))
    new_x1, padx1 = [new_x1_, 0] if new_x1_ > 0 else [0, -new_x1_]
    new_y1, pady1 = [new_y1_, 0] if new_y1_ > 0 else [0, -new_y1_]
    new_x2, padx2 = [new_x2_, 0] if new_x2_ < img_w else [img_w, new_x2_ - img_w]
    new_y2, pady2 = [new_y2_, 0] if new_y2_ < img_h else [img_h, new_y2_ - img_h]
    new_x1 = max(0, new_x1 - padx2)
    new_y1 = max(0, new_y1 - pady2)
    new_x2 = min(new_x2 + padx1, img_w)
    new_y2 = min(new_y2 + pady1, img_h)
    face_img = img[new_y1:new_y2, new_x1:new_x2]
    # 缺少像素补白边，保证人头在正中间
    # face_img = cv2.copyMakeBorder(face_img, pady1, pady2, padx1, padx2, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 半身图在底图中的坐标，用于之后计算人头,人脸在半身图中的坐标
    new_x1 = max(0, new_x1 - padx2)
    new_y1 = max(0, new_y1 - pady2)
    # new_x2 = min(new_x2 + padx1, img_w)
    # new_y2 = min(new_y2 + pady1, img_h)

    return face_img, (-new_x1, -new_y1)


## 增加图片亮度
def birght_img(img, ratio=1.5):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = np.power(img_yuv[:, :, 0] / 255, 1 / ratio) * 255
    equalize_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return equalize_img


## 将图片转成base64格式
def b64encode(img_cv2):
    # encode decode img
    img_encode = cv2.imencode('.jpg', img_cv2)[1]
    data_encode = np.array(img_encode)
    return base64.b64encode(data_encode).decode()


## 判断点是否在区域内
def is_point_in_area(point, area):
    flag = cv2.pointPolygonTest(area, point, False)
    if flag == 1:
        return True
    else:
        return False


## 判断框是否全部在区域内
def is_box_in_area(box, area):
    area_dict = {"xmin": int(area[0][0]), "ymin": int(area[0][1]), "xmax": int(area[2][0]), "ymax": int(area[2][1])}
    box_dict = {"xmin": int(box[0]), "ymin": int(box[1]), "xmax": int(box[0]) + int(box[2]),
                "ymax": int(box[1]) + int(box[3])}

    if box_dict["xmin"] > area_dict["xmin"] and box_dict["ymin"] > area_dict["ymin"] \
            and box_dict["xmax"] < area_dict["xmax"] and box_dict["ymax"] < area_dict["xmax"]:
        return True
    else:
        return False


def save_img2dir(img, file_name, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    cv2.imencode('.jpg', img)[1].tofile(os.path.join(path, file_name))


def save_json2dir(ann, info, json_file, width, height, offset):
    face_box = ann["face_box"]
    face_box = [int(face_box[0] + offset[0]), int(face_box[1] + offset[1]), int(face_box[2]), int(face_box[3])]

    head_box = ann["head_box"]
    head_box = [int(head_box[0] + offset[0]), int(head_box[1] + offset[1]), int(head_box[2]), int(head_box[3])]

    boxes = {"face": face_box, "head": head_box}

    trackId_uuid = str(info["uuid"]) + "_" + str(ann["track_id"]) + "_" + str(info["track_index"])

    person = {"trackId": int(ann["track_id"]), "age": int(ann["age"]), "gender": ann["gender"],
              "valid_track": info["valid_track"], "trackId_uuid": trackId_uuid,
              "score": ann["score"], "confidence": ann["face_confidence"], "picBox": "half",
              "yaw": int(ann["angle"][0]), "pitch": int(ann["angle"][1]),
              "roll": int(ann["angle"][2]), "boxes": boxes}

    pic = {"media_id": info["media_id"], "eventTime": int(time.time()), "width": width,
           "height": height, "person": person}

    with open(json_file, "w") as f:
        content = json.dumps(pic, indent=4)
        f.write(content)


def get_local_filename(info):
    return 'face' + "_".join((str(int(info["width"])), str(int(info["height"])), str(int(info["time"])),
                              str(int(info["track_id"])), str(int(info["score"] * 100)), str(info["image_id"])))


def save_person_info_to_local(img, track_info, extract_info, business_params_other):
    annotation_info = track_info["annotation_info"]

    head_box = annotation_info["head_box"]

    # 半身图
    face_pad_img, offset = get_pad_img(head_box, img)

    # 提高亮度
    if business_params_other["USE_BRIGHTNESS_ENHANCEMENT"]:
        face_pad_img = birght_img(face_pad_img, business_params_other["BRIGHTNESS_GAIN"])

    shape = face_pad_img.shape

    info = {
        "width": shape[0],
        "height": shape[1],
        "track_id": annotation_info["track_id"],
        "score": annotation_info["score"],
        "time": int(time.time()),
        "image_id": extract_info["image_id"]
    }

    face_name = get_local_filename(info)

    save_img2dir(face_pad_img, face_name + '.jpg', extract_info["dirname"] + '/')
    save_json2dir(annotation_info, extract_info, extract_info["dirname"] + '/' + face_name + '.json', shape[0],
                  shape[1], offset)


def save_pic_image(path, file, content):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path + file, 'wb') as f:
            f.write(base64.b64decode(content))
    except:
        print(traceback.format_exc())


def save_best_person_info_to_local(best_face_info, extract_info):
    annotation_info = best_face_info["person"]

    shape = best_face_info["shape"]
    offset = best_face_info["offset"]

    info = {
        "width": shape[0],
        "height": shape[1],
        "track_id": annotation_info["track_id"],
        "score": annotation_info["score"],
        "time": int(time.time()),
        "image_id": extract_info["image_id"]
    }

    face_name = get_local_filename(info)

    save_pic_image(extract_info["dirname"] + '/', face_name + '.jpg', best_face_info["picFile"])
    save_json2dir(annotation_info, extract_info, extract_info["dirname"] + '/' + face_name + '.json', shape[0],
                  shape[1], offset)


def get_bgpic_picinfo(best_bgpic_info):
    img_encode = cv2.imencode('.jpg', best_bgpic_info['best_img'])[1]
    height = best_bgpic_info['best_img'].shape[0]
    width = best_bgpic_info['best_img'].shape[1]
    data_encode = np.array(img_encode)
    encode_img = base64.b64encode(data_encode).decode()
    return encode_img, width, height
