import cv2

from modules.utils.colormap import colormap
from modules.utils.vis_utils import vis_key_points, vis_posebox, vis_age_gender


def box_track(img, track_id, head_box, track_points, face_box, business_params_draw):
    color_map = colormap()
    color = int(color_map[track_id % business_params_draw["MAX_COLOR_NUM"]][0]), int(
        color_map[track_id % business_params_draw["MAX_COLOR_NUM"]][1]), int(
        color_map[track_id % business_params_draw["MAX_COLOR_NUM"]][2])

    # track_points会包含起始点，如果长度>=DRAW_TRACK_NUM，则第一个点不画
    if business_params_draw["TRACK_DRAW_LESS"] and len(track_points) >= business_params_draw["DRAW_TRACK_NUM"]:
        draw_track = track_points[-business_params_draw["DRAW_TRACK_NUM"]:]
    else:
        draw_track = track_points[:]

    for j in range(len(draw_track)):
        if business_params_draw["DRAW_TRACK"]:
            cv2.circle(img, (draw_track[j][0], draw_track[j][1]), business_params_draw["TRACK_CIRCLE_RADIUS"], color,
                       business_params_draw["TRACK_CIRCLE_SIZE"])
        if business_params_draw["DRAW_HEAD"]:
            if j == len(draw_track) - 1:
                cv2.rectangle(img, (head_box[0], head_box[1]),
                              (head_box[0] + head_box[2], head_box[1] + head_box[3]),
                              color, thickness=business_params_draw["HEAD_RECTANGLE_SIZE"])
        if business_params_draw["DRAW_FACE"] and face_box:
            if j == len(draw_track) - 1:
                cv2.rectangle(img, (face_box[0], face_box[1]),
                              (face_box[0] + face_box[2], face_box[1] + face_box[3]),
                              color, thickness=business_params_draw["FACE_RECTANGLE_SIZE"])
        if business_params_draw["DRAW_TRACK"]:
            if j != 0:
                cv2.line(img, (draw_track[j - 1][0], draw_track[j - 1][1]),
                         (draw_track[j][0], draw_track[j][1]), color, thickness=business_params_draw["TRACK_LINE_SIZE"])
    return img


def customer_num(img, in_total, out_total, business_params_draw):
    if business_params_draw["DRAW_COUNT"]:
        cv2.putText(img, "IN :" + str(in_total), business_params_draw["COUNT_IN_ORG"], cv2.FONT_HERSHEY_SIMPLEX,
                    business_params_draw["COUNT_TEXT_FONT_SCALE"], business_params_draw["COUNT_IN_COLOR"],
                    business_params_draw["COUNT_TEXT_SIZE"])
        cv2.putText(img, "OUT:" + str(out_total), business_params_draw["COUNT_OUT_ORG"], cv2.FONT_HERSHEY_SIMPLEX,
                    business_params_draw["COUNT_TEXT_FONT_SCALE"], business_params_draw["COUNT_OUT_COLOR"],
                    business_params_draw["COUNT_TEXT_SIZE"])

    return img


def box_key_point(img, image_info, business_params_draw):
    if business_params_draw["DRAW_KEY_POINT"] or business_params_draw["DRAW_POSEBOX"]:
        color_map = colormap()
        for annotation_info in image_info["annotations"]:
            face_box = annotation_info["face_box"]
            head_box = annotation_info["head_box"]
            ps = annotation_info["ps"]
            key_points = annotation_info["key_points"]
            track_id = annotation_info["track_id"]

            color = int(color_map[track_id % business_params_draw["MAX_COLOR_NUM"]][0]), int(
                color_map[track_id % business_params_draw["MAX_COLOR_NUM"]][1]), int(
                color_map[track_id % business_params_draw["MAX_COLOR_NUM"]][2])

            # 绘制人头框
            if head_box and business_params_draw["DRAW_FACE"]:
                cv2.rectangle(img, (head_box[0], head_box[1]), (head_box[0] + head_box[2], head_box[1] + head_box[3]),
                              color, business_params_draw["HEAD_RECTANGLE_SIZE"])

            if face_box:
                # 绘制人脸框
                if business_params_draw["DRAW_FACE"]:
                    cv2.rectangle(img, (face_box[0], face_box[1]),
                                  (face_box[0] + face_box[2], face_box[1] + face_box[3]),
                                  color, business_params_draw["FACE_RECTANGLE_SIZE"])

                # 绘制3D关键点
                if business_params_draw["DRAW_KEY_POINT"]:
                    vis_key_points(img, key_points, business_params_draw)

                # 绘制3D框
                if business_params_draw["DRAW_POSEBOX"]:
                    vis_posebox(img, ps, key_points, color, business_params_draw)

    return img


def age_gender(img, image_info, business_params_draw):
    if business_params_draw["DRAW_AGE_GENDER"]:
        face_boxes = []
        infos = []

        for annotation_info in image_info["annotations"]:
            face_box = annotation_info["face_box"]

            # 绘制人脸框
            if face_box:

                gender = "男" if annotation_info["gender"] == "male" else "女"
                age = annotation_info["age"]
                if age < 21:
                    age = '小于20'
                elif age > 59:
                    age = '大于60'
                else:
                    age = str(int(age))
                info = gender + ' ' + age + '岁'

                face_boxes.append(face_box)
                infos.append(info)

        img = vis_age_gender(img, face_boxes, infos, business_params_draw)

    return img
