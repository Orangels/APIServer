from modules.models.model_deal.face_age_gender import get_age_gender_info
from modules.models.model_deal.face_key_point import get_face_key_point_info
from modules.models.model_deal.get_single_image_info import get_single_image_info


class ModelDeal:
    def __init__(self):
        pass

    # 检测
    def face_detection(self, bbox):
        face_bbox = []
        head_bbox = []
        for box in bbox:
            if box[5] == 2:
                face_bbox.append(box.tolist())
            else:
                head_bbox.append(box.tolist())

        return head_bbox, face_bbox

    # 跟踪
    def face_tracker(self, tracks, delete_tracks):
        track_ids = [track[0] for track in tracks]
        delete_track_ids = [track[0] for track in delete_tracks]
        return track_ids, delete_track_ids

    # 性别年龄
    def face_age_gender(self, age_gender):
        return get_age_gender_info(age_gender)

    # 3D关键点
    def face_key_point(self, kpts):
        return get_face_key_point_info(kpts)

    # 整个每个track的数据
    def get_image_info(self, image_id, face_bbox, head_bbox, track_ids, delete_track_ids, ages, genders, norms,
                       kpts_list, pss, angles):
        return get_single_image_info(image_id, face_bbox, head_bbox, track_ids, delete_track_ids, ages, genders, norms,
                                     kpts_list, pss, angles)
