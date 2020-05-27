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

    def get_result(self, img):
        try:
            self.num += 1
            ret = []
            frame = img
            if self.num == self.stop_num:
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


