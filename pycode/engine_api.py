import sys
import cv2
import time
import numpy as np
import traceback
import random

class ObjectApi():
    def __init__(self, cfg_file='detection.yaml'):
        self.num = 0
        self.stop_num = random.randint(5,10)
        self.serialize = True
        print("Init {}".format(self.__class__.__name__))

    def get_result(self, img, bbox, trackIDs, kpts, ageGender):
        try:
            # print('****')
            # print(bbox[0])
            # print(trackIDs[0])
            # print(kpts[0].shape)
            # print(ageGender[0])
            # print('****')
            self.num += 1
            ret = []
            frame = img[0]

            # if len(kpts[0]) > 0 and self.serialize:
            #     print(ageGender[0])
                # print(kpts[0])
                # np.save('./npy/bbox.npy', bbox[0])
                # np.save('./npy/kpts.npy', kpts[0])
                # np.save('./npy/trackIDs.npy', trackIDs[0])
                # np.save('./npy/ageGender.npy', ageGender[0])
                # cv2.imwrite("./npy/npy_img_{}.jpg".format(self.stop_num), frame)
                # self.serialize = False
                # print('save npy')
            #
            # if self.num == self.stop_num:
            #     print('write img')
            #     cv2.imwrite("./imgs/pyimg_{}.jpg".format(self.stop_num), frame)
            return ret
        except Exception as e:
            print('***********')
            print(e)
            traceback.print_exc()
            print('***********')
            return ret


    def __del__(self):
        print(self.__class__.__name__)


if __name__ == '__main__':
    api = ObjectApi()


