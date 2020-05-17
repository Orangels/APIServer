import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue, Pool, Manager

sys.path.append('../../Pet-engine')

if __name__ == '__main__':

    from modules import pet_engine
    #
    module = pet_engine.MODULES['ObjectDet']
    det = module(
        cfg_file='detection.yaml',
        cfg_list=[]
    )
    frame = cv2.imread('/home/user/workspace/xxs/DPH_Server/build/imgs/183.jpg')
    cut0 = frame[0:960, 1920:2870]  # h, w
    result = det(cut0)
    for label, det in zip(result["im_labels"],result["im_dets"]):
        if det[4] > 0.5:
            cv2.rectangle(cut0, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 1)
            print(label, det)
    cv2.imwrite('test.png', cut0)


    # video_name = '/home/wangzhihui/densepose/845.mp4'
    # save_name = 'test.avi'
    # cap = cv2.VideoCapture(video_name)
    # w = int(cap.get(3))
    # h = int(cap.get(4))
    # fps = cap.get(5)
    # videowriter = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (w, h))
    # num_images = int(cap.get(7))
    # for i in range(num_images):
    #     ret, img = cap.read()
    #     result = det(img)
    #     print result