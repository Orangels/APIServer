import cv2
import numpy as np

from modules.utils.business_utils import is_point_in_area


class CustomerCheck:

    def _rect_inter_inner(self, x1, x2):
        n1 = x1.shape[0] - 1
        n2 = x2.shape[0] - 1
        X1 = np.c_[x1[:-1], x1[1:]]
        X2 = np.c_[x2[:-1], x2[1:]]
        S1 = np.tile(X1.min(axis=1), (n2, 1)).T
        S2 = np.tile(X2.max(axis=1), (n1, 1))
        S3 = np.tile(X1.max(axis=1), (n2, 1)).T
        S4 = np.tile(X2.min(axis=1), (n1, 1))
        return S1, S2, S3, S4

    def _rectangle_intersection_(self, x1, y1, x2, y2):
        S1, S2, S3, S4 = self._rect_inter_inner(x1, x2)
        S5, S6, S7, S8 = self._rect_inter_inner(y1, y2)

        C1 = np.less_equal(S1, S2)
        C2 = np.greater_equal(S3, S4)
        C3 = np.less_equal(S5, S6)
        C4 = np.greater_equal(S7, S8)

        ii, jj = np.nonzero(C1 & C2 & C3 & C4)
        return ii, jj

    # 将点坐标列表转成numpy格式 x_points y_points
    # param [[123, 75], [1814, 75], [1814, 994], [123, 994], [123, 75]]
    # return ([ 123 1814 1814  123  123], [ 75  75 994 994  75])
    def transfer_np(self, points):
        np_points = np.array(points)
        x_points = np_points[:, 0]
        y_points = np_points[:, 1]
        return x_points, y_points

    # 将点坐标列表转成numpy格式 x_points y_points inv_x_points inv_y_points
    # inv_x_point为x_point的倒序，数据的顺序相反。表示进门线的反方向
    # param [[30, 546], [525, 797], [1490, 812], [1908, 449]]
    # return ([  30  525 1490 1908], [546 797 812 449], [1908 1490  525   30], [449 812 797 546])
    def transfer_np_and_inv(self, points):
        np_points = np.array(points)
        np_points_inv = np_points[::-1]
        x_points = np_points[:, 0]
        y_points = np_points[:, 1]
        inv_x_points = np_points_inv[:, 0]
        inv_y_points = np_points_inv[:, 1]
        return x_points, y_points, inv_x_points, inv_y_points

    # 计算 x1,y1 和 x2,y2的交集
    # params: [ 123 1814 1814  123  123] [ 75  75 994 994  75] [  30  525 1490 1908] [546 797 812 449]
    # return: （[1814.  123.], [530.63157895 593.15757576])
    def intersection(self, x1, y1, x2, y2):
        ii, jj = self._rectangle_intersection_(x1, y1, x2, y2)
        n = len(ii)

        dxy1 = np.diff(np.c_[x1, y1], axis=0)
        dxy2 = np.diff(np.c_[x2, y2], axis=0)

        T = np.zeros((4, n))
        AA = np.zeros((4, 4, n))
        AA[0:2, 2, :] = -1
        AA[2:4, 3, :] = -1
        AA[0::2, 0, :] = dxy1[ii, :].T
        AA[1::2, 1, :] = dxy2[jj, :].T

        BB = np.zeros((4, n))
        BB[0, :] = -x1[ii].ravel()
        BB[1, :] = -x2[jj].ravel()
        BB[2, :] = -y1[ii].ravel()
        BB[3, :] = -y2[jj].ravel()

        for i in range(n):
            try:
                T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
            except:
                T[:, i] = np.NaN

        in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

        xy0 = T[2:, in_range]
        xy0 = xy0.T
        return xy0[:, 0], xy0[:, 1]

    def transfer_mergexy(self, points_x, points_y):
        if points_x[0] < points_x[1]:
            return np.dstack((points_x, points_y))
        else:
            points_x = np.array([points_x[1], points_x[0]])
            points_y = np.array([points_y[1], points_y[0]])
            return np.dstack((points_x, points_y))

    def check_position_case(self, v_cross_points, v_entrance_line, v_rect):
        # determine the type of uncertain case
        """
        :param v_cross_points:
        :param v_entrance_line:
        :param v_rect: rectangle area positions (clockwise)
        :return: case_type, entrance_line, cross_points
        case_type: -1, bad case
                    1, two_cross_points both in one side of rectangle
                    2, only one segment of entranceline cross the whole rectangle
         entrance_line: entrance_line in order
         cross_points:  cross_points in order
        """
        # the num of entrance_line points minus one
        num_entrance_seg = len(v_entrance_line) - 1
        # the num of rectangle area points
        num_rect_seg = len(v_rect) - 1

        # record whether the entrance_segment has cross_point  key: index of entrance points ,value: boolean whether has
        # cross_point in this segment
        on_entrance_seg = {}
        # record whether the rectangle_segment has cross_point  key: index of rectangle points ,value: boolean whether
        # has cross_point in this segment
        on_rect_seg = {}
        # record cross_point coresponding to which rectangle_segment key: index of cross points ,value: index of
        # entrance_segment points
        on_cross_seg = {}
        # record cross_point coresponding to which entrance_segment key: index of cross points, value:
        # entrance points
        on_rect_cross_seg = {}

        for idx, v_cross in enumerate(v_cross_points):
            for i in range(num_entrance_seg):
                if i not in on_entrance_seg:
                    on_entrance_seg[i] = False
                if self.point_on_line_check(v_entrance_line[i], v_entrance_line[i + 1], v_cross):
                    on_entrance_seg[i] = True
                if on_entrance_seg[i]:
                    if idx not in on_cross_seg:
                        on_cross_seg[idx] = i

        for idx, v_cross in enumerate(v_cross_points):
            for i in range(num_rect_seg):
                if i not in on_rect_seg:
                    on_rect_seg[i] = False
                if self.point_on_line_check(v_rect[i], v_rect[i + 1], v_cross):
                    on_rect_seg[i] = True
                if on_rect_seg[i]:
                    on_rect_cross_seg[idx] = i

        # how many rect_segment have cross point
        count_rect = 0
        # how many entra_segment have cross point
        count_entra = 0
        # index of the entrance_segment which has cross point
        index_of_entra = []
        # index of the entrance_segment which has cross point
        index_of_rect = []

        for k, v in on_rect_seg.items():
            if v is True:
                count_rect += 1
                index_of_rect.append(k)

        for k, v in on_entrance_seg.items():
            if v is True:
                count_entra += 1
                index_of_entra.append(k)

        # in case 1, correct the order of cross_points and entrance_line
        # index_of_rect only has one value representing cross_points on which rect_seg
        # just for correcting the order of v_cross_points
        if count_rect == 1:
            if index_of_rect[0] == 0:
                # cross top line
                if v_cross_points[0][0] > v_cross_points[1][0]:
                    v_cross_points = v_cross_points[::-1]

            if index_of_rect[0] == 1:
                # cross right line
                if v_cross_points[0][1] > v_cross_points[1][1]:
                    v_cross_points = v_cross_points[::-1]

            if index_of_rect[0] == 2:
                # cross bottom line
                if v_cross_points[0][0] > v_cross_points[1][0]:
                    pass
                else:
                    v_cross_points = v_cross_points[::-1]

            if index_of_rect[0] == 3:
                # cross left line
                if v_cross_points[0][1] > v_cross_points[1][1]:
                    pass
                else:
                    v_cross_points = v_cross_points[::-1]

            # correct the order of entrance_line
            seg_order = []
            for idx in range(2):
                for i in range(len(v_entrance_line) - 1):
                    if self.point_on_line_check(v_entrance_line[i], v_entrance_line[i + 1], v_cross_points[idx]):
                        seg_order.append(i)

            if seg_order[0] > seg_order[1]:
                v_entrance_line = v_entrance_line[::-1]

            return 1, v_entrance_line, v_cross_points

        # in case 2, correct order of cross_points
        # just only to correct the order of cross_points for two cross points in one entrance_segment
        if count_entra == 1:
            if on_cross_seg[0] > on_cross_seg[1]:
                v_cross_points = v_cross_points[::-1]
            return 2, v_entrance_line, v_cross_points

        return -1, None, None

    def point_on_line_check(self, start_point, end_point, input_point):
        # check point whether in line segment (between start_point and end_point)
        if np.array_equal(start_point, end_point):
            return False
        vector_input2start = input_point - start_point  # vector input_point -> vector start_point
        vector_input2end = input_point - end_point  # vector input_point -> vector end_point
        flag_on_line = np.linalg.norm(np.cross(vector_input2start, vector_input2end))
        if round(flag_on_line, 4) == 0.0000:
            if np.dot(vector_input2start, vector_input2end) > 0:
                return False
            else:
                return True
        else:
            return False

    '''
        根据识别区域、进门方向、进门线获取进/出区域,返回构成进出区域多边形 点的坐标
        进门线将识别区域分成两个区域，通过进门方向决定进区域/出区域
        params: rect [[123, 75], [1814, 994]]
        params: direction [[983, 147], [998, 937]]
        params: entrance_line [[30, 546], [525, 797], [1490, 812], [1908, 449]]
        return:
            out_area: [[ 123.       75.    ]
             [1814.       75.    ]
             [1814.      530.6316]
             [1490.      812.    ]
             [ 525.      797.    ]
             [ 123.      593.1576]]
             
            in_area: [[ 123.      593.1576]
             [ 525.      797.    ]
             [1490.      812.    ]
             [1814.      530.6316]
             [1814.      994.    ]
             [ 123.      994.    ]]
         
    '''

    def find_inout_area(self, rect, direction, entrance_line):
        # 将识别区域转成numpy格式 give rectangle_line calculating format
        x1, y1 = self.transfer_np([[rect[0][0], rect[0][1]], [rect[1][0], rect[0][1]], [rect[1][0], rect[1][1]],
                                   [rect[0][0], rect[1][1]], [rect[0][0], rect[0][1]]])

        # 将进门线转成numpy格式 give entrance_line calculating format and its reverse
        x2, y2, x2_inv, y2_inv = self.transfer_np_and_inv(entrance_line)

        # 计算识别区域和进门线的交集 calculate the intersection points
        ret_x, ret_y = self.intersection(x1, y1, x2, y2)

        # must have two cross points
        if len(ret_x) != 2:
            # print("exit1")
            return None, None

        # check which case to run
        ret_x_, ret_y_ = self.intersection(x1, y1, x2_inv, y2_inv)
        ret_x_inv, ret_y_inv = self.intersection(x2, y2, x1, y1)
        ret_x_inv_, ret_y_inv_ = self.intersection(x2_inv, y2_inv, x1, y1)
        np_rect = np.array([[rect[0][0], rect[0][1]], [rect[1][0], rect[0][1]], [rect[1][0], rect[1][1]],
                            [rect[0][0], rect[1][1]], [rect[0][0], rect[0][1]]])

        flag_ret = all(np.equal(ret_x, ret_x_inv)) and all(np.equal(ret_y, ret_y_inv))
        flag_ret_inv = all(np.equal(ret_x_, ret_x_inv_)) and all(np.equal(ret_y_, ret_y_inv_))

        if flag_ret and flag_ret_inv:
            np_entrance_line = np.array(entrance_line)
            np_cross_points = self.transfer_mergexy(ret_x, ret_y)
            case_flag, np_entrance_line, np_cross_points[0] = self.check_position_case(np_cross_points[0],
                                                                                       np_entrance_line, np_rect)
            # run case -1,1,2
            if case_flag == -1:
                # print("exit2")
                return None, None

            area1 = []
            area2 = []
            if case_flag == 1:
                for idx in range(len(np_rect) - 1):
                    la = np_rect[idx]
                    lb = np_rect[idx + 1]
                    if self.point_on_line_check(la, lb, np_cross_points[0][0]):
                        area1.append(la)
                        area1.append(np_cross_points[0][0])
                        area2.append(np_cross_points[0][0])

                        cp_flag = False
                        for idx_ in range(len(np_entrance_line) - 1):
                            la_ = np_entrance_line[idx_]
                            lb_ = np_entrance_line[idx_ + 1]

                            if self.point_on_line_check(la_, lb_, np_cross_points[0][0]):
                                cp_flag = True
                                area1.append(lb_)
                                area2.append(lb_)
                                continue

                            if self.point_on_line_check(la_, lb_, np_cross_points[0][1]):
                                cp_flag = False
                                break

                            if cp_flag:
                                area1.append(lb_)
                                area2.append(lb_)
                                continue

                        area1.append(np_cross_points[0][1])
                        area2.append(np_cross_points[0][1])
                    else:
                        area1.append(la)

            if case_flag == 2:
                v_rect = np.array([[rect[0][0], rect[0][1]], [rect[1][0], rect[0][1]], [rect[1][0], rect[1][1]],
                                   [rect[0][0], rect[1][1]], [rect[0][0], rect[0][1]]])

                cp_seg_flag = []
                for idx in range(2):
                    for i in range(4):
                        la, lb = v_rect[i], v_rect[i + 1]
                        if self.point_on_line_check(la, lb, np_cross_points[0][idx]):
                            cp_seg_flag.append(i)

                if cp_seg_flag[0] > cp_seg_flag[1]:
                    np_cross_points[0] = np_cross_points[0][::-1]

                cp_flag = False
                for idx in range(len(np_rect) - 1):
                    la = np_rect[idx]
                    lb = np_rect[idx + 1]

                    flag_on_seg1 = self.point_on_line_check(la, lb, np_cross_points[0][0])
                    flag_on_seg2 = self.point_on_line_check(la, lb, np_cross_points[0][1])
                    flag_on_seg = flag_on_seg2 or flag_on_seg1

                    if flag_on_seg:
                        if cp_flag:
                            cp_flag = False
                        else:
                            cp_flag = True

                    if not flag_on_seg and not cp_flag:
                        area1.append(la)

                    if flag_on_seg and cp_flag:
                        area1.append(la)
                        area1.append(np_cross_points[0][0])
                        area1.append(np_cross_points[0][1])
                        area2.append(np_cross_points[0][1])
                        area2.append(np_cross_points[0][0])

                    if not flag_on_seg and cp_flag:
                        area2.append(la)

                    if flag_on_seg and not cp_flag:
                        area2.append(la)

        else:
            # run case normal cross points in two opposite rectangle segment
            x1, y1 = self.transfer_np([[rect[0][0], rect[0][1]], [rect[1][0], rect[0][1]], [rect[1][0], rect[1][1]],
                                       [rect[0][0], rect[1][1]], [rect[0][0], rect[0][1]]])
            x2, y2 = self.transfer_np(entrance_line[::-1])
            ret_x, ret_y = self.intersection(x1, y1, x2, y2)
            np_entrance_line = np.array(entrance_line[::-1])

            np_cross_points = self.transfer_mergexy(ret_x, ret_y)

            v_rect = np.array([[rect[0][0], rect[0][1]], [rect[1][0], rect[0][1]], [rect[1][0], rect[1][1]],
                               [rect[0][0], rect[1][1]], [rect[0][0], rect[0][1]]])

            seg_flag = []
            for idx in range(len(np_entrance_line) - 1):
                for i in range(4):
                    seg_x, seg_y = self.transfer_np([np_entrance_line[idx], np_entrance_line[idx + 1]])
                    rec_x, rect_y = self.transfer_np([v_rect[i], v_rect[i + 1]])
                    ret_xx, ret_yy = self.intersection(seg_x, seg_y, rec_x, rect_y)
                    if len(ret_xx) == 1:
                        seg_flag.append(i)

            if seg_flag[0] > seg_flag[1]:
                np_entrance_line = np_entrance_line[::-1]

            cp_seg_flag = []
            for idx in range(2):
                for i in range(4):
                    la, lb = v_rect[i], v_rect[i + 1]
                    if self.point_on_line_check(la, lb, np_cross_points[0][idx]):
                        cp_seg_flag.append(i)

            if cp_seg_flag[0] > cp_seg_flag[1]:
                np_cross_points[0] = np_cross_points[0][::-1]

            # find cross_point segment in entrance_line
            seg = []
            for cross_point in np_cross_points[0]:
                for idx in range(len(np_entrance_line) - 1):
                    la = np_entrance_line[idx]
                    lb = np_entrance_line[idx + 1]
                    flag_on_seg = self.point_on_line_check(la, lb, cross_point)
                    if flag_on_seg:
                        seg.append([cross_point, [la, lb]])

            if len(seg) == 0:
                # print("exit3")
                return None, None

            # find the line connect two cross_points
            seg_line = []
            if np.array_equal(seg[0][1][0], seg[1][1][0]) and np.array_equal(seg[0][1][1], seg[1][1][1]):
                seg_line.append(seg[0][0])
                seg_line.append(seg[1][0])
            else:
                append_flag = False
                for idx in range(len(np_entrance_line) - 1):
                    la = np_entrance_line[idx]
                    lb = np_entrance_line[idx + 1]
                    if np.array_equal(seg[0][1][0], la) and np.array_equal(seg[0][1][1], lb) and (not append_flag):
                        seg_line.append(seg[0][0])
                        append_flag = True
                        continue

                    if np.array_equal(seg[1][1][0], la) and np.array_equal(seg[1][1][1], lb) and append_flag:
                        seg_line.append(la)
                        seg_line.append(seg[1][0])
                        break

                    if append_flag:
                        seg_line.append(la)
                        continue

            # gather two areas
            area1 = []
            area2 = []
            cp_flag = False
            for idx in range(len(np_rect) - 1):
                la = np_rect[idx]
                lb = np_rect[idx + 1]
                flag_on_seg = False

                flag_on_seg1 = self.point_on_line_check(la, lb, seg[0][0])
                flag_on_seg2 = self.point_on_line_check(la, lb, seg[1][0])
                if flag_on_seg1 or flag_on_seg2:
                    flag_on_seg = True

                if flag_on_seg:
                    if cp_flag:
                        cp_flag = False
                    else:
                        cp_flag = True

                if not flag_on_seg and not cp_flag:
                    area1.append(la)

                if flag_on_seg and cp_flag:
                    area1.append(la)
                    area1 = area1 + seg_line
                    area2 = area2 + seg_line[::-1]

                if not flag_on_seg and cp_flag:
                    area2.append(la)

                if flag_on_seg and not cp_flag:
                    area2.append(la)

        # print("-----print area------")
        # print("-----area1-----")
        # print(area1)
        # print("-----area2-----")
        # print(area2)
        area1 = np.array(area1).astype(np.float32)
        area2 = np.array(area2).astype(np.float32)

        # find which one in in_area and which one in out_area
        in_point = (direction[0][0], direction[0][1])
        out_point = (direction[1][0], direction[1][1])
        flag = cv2.pointPolygonTest(area1, in_point, False)
        flag_ = cv2.pointPolygonTest(area1, out_point, False)

        if flag == 1:
            if flag_ == 1:
                # print("exit4")
                return None, None
            else:
                # print("exit5")
                return area1, area2
        else:
            # print("exit6")
            return area2, area1

    ## 根据相对坐标和图片宽高获取真实坐标
    def get_cordinate(self, cor_obj, shape):
        num = len(cor_obj)
        if num == 0:
            return None
        cordinate = []
        for idx in range(num):
            tmp = [int(cor_obj[idx]["x"] * shape[1]), int(cor_obj[idx]["y"] * shape[0])]
            cordinate.append(tmp)
        return cordinate

    ## 根据轨迹+进区域+出区域，判断进店或是出店
    def deter_in_out(self, track, business_params):

        business_params_count = business_params["COUNT"]
        direction = business_params_count["ENTRANCE_DIRECTION"]
        in_area = business_params["DEAL"]["IN_AREA"]
        out_area = business_params["DEAL"]["OUT_AREA"]

        # 起始点和终点
        start_point = tuple(track[0])
        end_point = tuple(track[-1])

        # 没画进门线，或是进门线和识别区域没有形成有效进出时，默认为符合条件
        if in_area is None and out_area is None:
            pass
        else:
            if is_point_in_area(start_point, out_area) and is_point_in_area(end_point, in_area):
                pass
                print("start in out,end in in...")
            elif is_point_in_area(end_point, out_area) and is_point_in_area(start_point, in_area):
                pass
                print("start in in,end in out...")
            else:
                ## 起始点和终点在用一个区域内，不是有效的轨迹，非进非出，返回0,0
                print('type 1: start_point not in out_area or end_point not in in_area')
                return 0, 0

        x_end = end_point[0]
        y_end = end_point[1]

        # 计算人起始点和终点的长度
        vect_end = np.array([x_end, y_end])
        vect_start = np.array([start_point[0], start_point[1]])
        vect_person = vect_start - vect_end
        dist_vect = np.sqrt(vect_person.dot(vect_person))

        # 计算进门方向(ENTRANCE_DIRECTION)起始点和终点的长度
        direction_start = np.array([direction[0][0], direction[0][1]])
        direction_end = np.array([direction[1][0], direction[1][1]])
        vect_dir = direction_start - direction_end
        dist_dir = np.sqrt(vect_dir.dot(vect_dir))

        # 人的长度小于进门方向长度的VECTOR_LEN，不是有效的轨迹，非进非出，返回0,0
        if dist_vect / dist_dir < business_params_count["VECTOR_LEN"]:
            print('type 2: trace vector is not long enough')
            return 0, 0
        else:
            # 根据人起始点终点连线和进门方向的夹角判断进和出
            angle = np.arccos(1.0 * vect_person.dot(vect_dir / (dist_dir * dist_vect)))
            angle = angle * 360 / 2 / np.pi

            # 夹角在0到ANGLE之间为进
            if 0 <= angle <= business_params_count["ANGLE"]:
                # print('type 3: trace in')
                return 1, 0

            # 夹角在180-ANGLE到180之间为出
            if 180 - business_params_count["ANGLE"] <= angle <= 180:
                print('type 4: trace out')
                return 0, 1
            print('type 5: trace invalid')
            return 0, 0

    # 计算识别区域的左上和右下坐标
    def find_rect(self, points, shape):
        p_array = np.array(points)
        x = p_array[:, 0]
        y = p_array[:, 1]

        xmax, xmin = np.max(x), np.min(x)
        ymax, ymin = np.max(y), np.min(y)

        if xmin < 0:
            xmin = 0
        if xmax >= shape[1]:
            xmax = shape[1]
        if ymin < 0:
            ymin = 0
        if ymax >= shape[0]:
            ymax = shape[0]
        rect_out = [[xmin, ymin], [xmax, ymax]]
        return rect_out

    # 计算进门方向的起止坐标
    def find_direct(self, rect):
        x = int((rect[0][0] + rect[1][0]) / 2)
        y1 = rect[0][1] + int((rect[1][1] - rect[0][1]) / 4)
        y2 = rect[1][1] - int((rect[1][1] - rect[0][1]) / 4)
        direction_out = [[x, y1], [x, y2]]
        return direction_out


'''
{"roi_area":[{"x":0.06446,"y":0.069659},{"x":0.945122,"y":0.069659},{"x":0.945122,"y":0.921053},{"x":0.06446,"y":0.921053}],
"face_roi_area":[{"x":0.238676,"y":0.26935},{"x":0.779617,"y":0.26935},{"x":0.779617,"y":0.80805},{"x":0.238676,"y":0.80805}],
"entrance_line":[{"x":0.015679,"y":0.506192},{"x":0.268293,"y":0.73839},{"x":0.760453,"y":0.752322},{"x":0.973868,"y":0.416409}],
"entrance_direction":[{"x":0.501742,"y":0.136223},{"x":0.509582,"y":0.868421}]}

rect [[123, 75], [1814, 994]]
direction [[963, 147], [978, 937]]
entrance_line [[30, 546], [515, 797], [1460, 812], [1869, 449]]
out_area [[  123.            75.        ]
 [ 1814.            75.        ]
 [ 1814.           497.81417847]
 [ 1460.           812.        ]
 [  515.           797.        ]
 [  123.           594.12988281]]
in_area [[  123.           594.12988281]
 [  515.           797.        ]
 [ 1460.           812.        ]
 [ 1814.           497.81417847]
 [ 1814.           994.        ]
 [  123.           994.        ]]


'''

if __name__ == "__main__":
    customer_check = CustomerCheck()

    rect = [[123, 75], [1814, 994]]
    shape = [1080, 1920]

    direction = customer_check.get_cordinate([{"x": 0.501742, "y": 0.136223}, {"x": 0.509582, "y": 0.868421}], shape)
    entrance_line = customer_check.get_cordinate(
        [{"x": 0.015679, "y": 0.506192}, {"x": 0.268293, "y": 0.73839}, {"x": 0.760453, "y": 0.752322},
         {"x": 0.973868, "y": 0.416409}], shape)
    print(entrance_line)

    rect = [[0, 0], [1900, 1000]]
    entrance_line = [[0, 546], [525, 797], [1490, 812], [1910, 449]]

    out_area, in_area = customer_check.find_inout_area(rect, direction, entrance_line)

    print("rect:", rect)
    print("entrance_line:", entrance_line)
    print("direction:", direction)
    print("out_area:", out_area)
    print("in_area:", in_area)
