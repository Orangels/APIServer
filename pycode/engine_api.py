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
        print("Init {}".format(self.__class__.__name__))

    def get_result(self, img, bbox, kpts, ageGender):
        try:
            # print('****')
            # print(bbox[0])
            # print(kpts[0].shape)
            # print(ageGender[0].shape)
            # print('****')
            self.num += 1
            ret = []
            frame = img[0]
            if self.num == self.stop_num:
                print('write img')
                cv2.imwrite("./imgs/pyimg_{}.jpg".format(self.stop_num), frame)
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


