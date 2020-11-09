import platform
from math import sqrt

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def vis_key_points(img, key_points, business_params_draw):
    for j in range(key_points.shape[1]):
        cv2.circle(img, (int(round(key_points[0, j])), int(round(key_points[1, j]))),
                   business_params_draw["KEY_POINT_CIRCLE_RADIUS"], business_params_draw["KEY_POINT_COLOR"],
                   business_params_draw["KEY_POINT_CIRCLE_SIZE"])

    nums = [0, 16, 21, 26, 30, 35, 41, 47, 59, 67]

    # close eyes and mouths
    plot_close = lambda i1, i2: cv2.line(img, (int(round(key_points[0, i1])), int(round(key_points[1, i1]))),
                                         (int(round(key_points[0, i2])), int(round(key_points[1, i2]))),
                                         business_params_draw["KEY_POINT_COLOR"],
                                         business_params_draw["KEY_POINT_LINE_SIZE"])
    plot_close(41, 36)
    plot_close(47, 42)
    plot_close(59, 48)
    plot_close(67, 60)
    plot_close(36, 39)
    plot_close(42, 45)
    plot_close(48, 54)

    for ind in range(len(nums) - 1):
        if ind != 0:
            l, r = nums[ind] + 1, nums[ind + 1]
        else:
            l, r = nums[ind], nums[ind + 1]
        for k in range(l, r):
            cv2.line(img, (int(round(key_points[0, k])), int(round(key_points[1, k]))),
                     (int(round(key_points[0, k + 1])), int(round(key_points[1, k + 1]))),
                     business_params_draw["KEY_POINT_COLOR"], business_params_draw["KEY_POINT_LINE_SIZE"])


def vis_posebox(img, Ps, key_points, color, business_params_draw):
    pts68 = key_points
    llength = calc_hypotenuse(pts68)
    point_3d = build_camera_box(llength)
    P = Ps

    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(pts68[:2, :27], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, business_params_draw["POSEBOX_POLYLINES_SIZE"], cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(point_2d[6]), color, business_params_draw["POSEBOX_LINE_SIZE"], cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(point_2d[7]), color, business_params_draw["POSEBOX_LINE_SIZE"], cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(point_2d[8]), color, business_params_draw["POSEBOX_LINE_SIZE"], cv2.LINE_AA)


def vis_age_gender(img, face_boxes, infos, business_params_draw):
    """Visualizes the text."""
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    # vis time
    if platform.system() == "Windows":
        font = ImageFont.truetype(business_params_draw["AGE_GENDER_FONT"]["Windows"],
                                  business_params_draw["AGE_GENDER_FONT_SIZE"], encoding="utf-8")
    else:
        font = ImageFont.truetype(business_params_draw["AGE_GENDER_FONT"]["Linux"],
                                  business_params_draw["AGE_GENDER_FONT_SIZE"], encoding="utf-8")

    for i in range(len(infos)):
        class_str = infos[i]
        if not class_str == '':
            pos = face_boxes[i]
            x, y, w, h = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])

            # Compute text size.
            txt_w, txt_h = font.getsize(class_str)
            x0 = int(x + w / 2 - txt_w / 2)
            if y > txt_h:
                y0 = y - txt_h
            else:
                y0 = y + h

            # Show text.
            txt_tl = x0, y0
            draw.text(txt_tl, class_str, font=font, fill=business_params_draw["AGE_GENDER_COLOR"])
    img = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)

    return img


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    return point_3d
