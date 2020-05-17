import sys
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Pool, Manager
sys.path.append('/home/user/workspace/xxs/ENGIENE/Pet-engine')
from modules import pet_engine

import traceback

class ObjectApi():
    def __init__(self, cfg_file='detection.yaml'):
        module = pet_engine.MODULES['ObjectDet']
        self.det = module(cfg_list=['MODULES.OBJDET.CFG', 'ckpts/fcos-imprv_V-39-FPN-P5_1x/fcos-imprv_V-39-FPN-P5_1x.yaml'])
        # self.img = cv2.imread('/home/user/workspace/xxs/DPH_Server_tmp/pycode/test.png')
        # self.img = cv2.resize(self.img, (480, 960))

    def __call__(self):
        result = self.det(self.img)
        print('infer done')
        print(result)
        # return result

    def get_result(self, img, img1, img2, img3):
        try:
            ret = []
            print('get_result')
            print(img[0].shape)
            print(img1[0].shape)
            print(img2[0].shape)
            print(img3[0].shape)
            # return ret
            # result = self.det([img, img1, img2, img3])
            # print(result)
            return ret
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
            print(ret)
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
            result = self.det([self.img])
            print(result)
        ed = time.time()
        print('time test 100 times avg = ', (ed - s1)/100)



if __name__ == '__main__':
    api = ObjectApi()
    img = cv2.imread('/home/user/workspace/xxs/DPH_Server/data/test.png')
    api.get_result(img,img,img,img)
    # api.time_test()

