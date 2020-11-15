import sys
import cv2
import time
import numpy as np
import traceback
import random

class ObjectApi():
    def __init__(self, camId=0):
        self.num = 0
        self.stop_num = random.randint(5,10)
        self.serialize = True
        print("Init {} -- {}".format(self.__class__.__name__, camId))

    def get_result(self, img, bbox, trackIDs, deleteIDs, kpts, ageGender):
        try:
            # print('**py**')
            # print("*****")
            # print(bbox[0])
            # print(trackIDs[0])
            # print("*****")
            # print(kpts[0].shape)
            # print(ageGender[0])
            # print("delete ids : ")
            # print(deleteIDs[0])
            # print('**py**')
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
            # frame=cv2.rectangle(frame, (0, 0), (100, 100), (0,255,0), 2)

            return frame
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


