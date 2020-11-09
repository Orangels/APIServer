from modules.utils.frame_utils import box_track, customer_num, box_key_point, age_gender


class FrameDraw:
    def __init__(self):
        pass

    # 人脸框和轨迹
    def draw_box_track(self, img, track_ids, track_infos, business_params_draw):

        for track_id in track_ids:
            if track_id in track_infos and track_infos[track_id]["track_info"].get("in_area"):
                img = box_track(img, track_id, track_infos[track_id]["annotation_info"]["head_box"],
                                track_infos[track_id]["track_info"]["track_points"],
                                track_infos[track_id]["annotation_info"]["face_box"], business_params_draw)

        return img

    # 人脸框、人头框、3D关键点，跟框线无关，有检测到就绘制
    def draw_box_key_point(self, img, image_info, business_params_draw):
        return box_key_point(img, image_info, business_params_draw)

    # 性别年龄，跟框线无关，有检测到就绘制
    def draw_age_gender(self, img, image_info, business_params_draw):
        return age_gender(img, image_info, business_params_draw)

    # 进出店人次
    def draw_customer_num(self, img, in_total, out_total, business_params_draw):
        return customer_num(img, in_total, out_total, business_params_draw)
