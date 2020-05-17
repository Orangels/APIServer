import sys
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Pool, Manager
sys.path.append('/home/user/workspace/xxs/Pet-engine')
from modules import pet_engine
from projects.fh_tracking.fhtracker import HFtracker

import traceback

class ObjectApi():
    def __init__(self, cfg_file='detection.yaml'):
        module = pet_engine.MODULES['ObjectDet']
        self.det = module(
            cfg_file='/home/user/workspace/xxs/DPH_Server/pycode/detection.yaml',
            cfg_list=[]
        )
        # self.img = cv2.imread('/home/user/workspace/xxs/DPH_Server_tmp/pycode/test.png')
        # self.img = cv2.resize(self.img, (480, 960))

    def __call__(self):
        result = self.det(self.img)
        print('infer done')
        print(result)
        # return result

    def get_result(self, img):
        try:
            ret = []
            frame = img
            # return ret
            result = self.det(frame)
            if len(result["im_labels"]) == 0:
                return ret
            for label, det in zip(result["im_labels"], result["im_dets"]):
                if det[4] > 0.5:
                    ret.append(label)
                    ret.append(int(det[4]*100))
                    ret.append(int(det[0]))
                    ret.append(int(det[1]))
                    ret.append(int(det[2]))
                    ret.append(int(det[3]))
            # print(ret)
            return ret
        except Exception as e:
            print('***********')
            print(e)
            traceback.print_exc()
            print('***********')
            return ret

    def time_test(self):
        s1 = time.time()
        for i in range(100):
            result = self.det(self.img)
        ed = time.time()
        print('time test 100 times avg = ', (ed - s1)/100)



if __name__ == '__main__':
    api = ObjectApi()
    api.time_test()

