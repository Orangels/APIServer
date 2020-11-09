import numpy as np


def get_face_key_point_info(kpts):
    all_pts = []
    all_P = []
    all_popse = []
    ldmk_bboxes_index = 0

    ots = kpts.reshape(-1)
    numBox = len(kpts)
    for i in range(numBox):
        offset = 62 * numBox + i * 219
        pts68 = ots[offset:offset + 204].reshape([3, -1])

        ldmk_bboxes_index += 1
        offset += 204
        P = ots[offset:offset + 12].reshape([3, -1])
        offset += 12
        pose = ots[offset:offset + 3]
        all_pts.append(pts68)
        all_P.append(P)
        all_popse.append(pose.tolist())

    return all_pts, np.asarray(all_P), all_popse
