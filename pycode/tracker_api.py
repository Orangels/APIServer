import os
import sys
import argparse
import cv2
import torch
import time
sys.path.append('/home/jwh/workspace/Pet-dev')

from pet.projects.fairmot.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from pet.utils.net import convert_bn2affine_model
from pet.projects.fairmot.core.inference import inference
from pet.projects.fairmot.datasets.transform import build_transforms
from pet.projects.fairmot.modeling.model_builder import Generalized_CNN
from pet.utils.checkpointer import get_weights, load_weights
from pet.projects.fairmot.tracking.multitracker import JDETracker
from pet.projects.fairmot.utils.visualization import plot_tracking
from pet.utils.misc import logging_rank, mkdir_p

parser = argparse.ArgumentParser(description='Pet Model Evaluating')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='/home/jwh/workspace/Pet-dev/cfgs/projects/fairmot/mot/fairmot-DLA34.yaml', type=str)
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id for evaluation')
parser.add_argument('--cam', help='whether use webcam, default True, False to load video', type=bool)
parser.add_argument('--video_path', help='whether use webcam, default True, False to load video', type=str)
parser.add_argument('--output_path', help='path to save visualized img', type=str)
parser.add_argument('opts', help='See pet/project/higherhrnet/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def initialize_model_from_cfg():
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = Generalized_CNN(is_train=False)
    # Load trained model
    # cfg.TEST.WEIGHTS = get_weights(cfg.CKPT, cfg.TEST.WEIGHTS)
    cfg.TEST.WEIGHTS = '/home/jwh/workspace/Pet-dev/ckpts/projects/fairmot/mot/fairmot-DLA34/all_dla34_trans-model-pet-merge.pth'  # hard code
    load_weights(model, cfg.TEST.WEIGHTS)
    model.eval()
    model.to(torch.device(cfg.DEVICE))

    return model

class ObjectApi:
    def __init__(self, frame_rate=25, cfg_file=None, opts=[]):
        if args.cfg_file is not None:
            merge_cfg_from_file(args.cfg_file)
        if args.opts is not None:
            merge_cfg_from_list(opts)
        assert_and_infer_cfg(make_immutable=False)
        self.transforms = build_transforms(is_train=False)
        self.model0 = initialize_model_from_cfg()
        self.tracker0 = JDETracker(self.model0, frame_rate)  # frame rate???

    def __call__(self, img0, img1, img2, img3):

        vis_im = plot_tracking(img, tlwhs, ids)

        return dict(im_dets=tlwhs, im_ids=ids)

    def __del__(self):
        print(self.__class__.__name__)
        

    def get_result(self, img0):
        ret = []
        # print('get_result0', ret)
        tlwhs, ids = inference(self.tracker0, img0)
        # vis_im = plot_tracking(img0, tlwhs, ids)
        # current = time.time()
        # cv2.imwrite('./debug/0/' + str(int(current*10000000)) + '.png', vis_im)

        if len(tlwhs) == 0:
            return ret
        for label, det in zip(ids, tlwhs):
            ret.append(label)
            ret.append(0)
            ret.append(int(det[0]))
            ret.append(int(det[1]))
            ret.append(int(det[0]) + int(det[2]))
            ret.append(int(det[1]) + int(det[3]))
        # print('get_result0', ret)
        return ret






# using example
# def main():
#     # read img
#     if args.cam:
#         cap = cv2.VideoCapture(0)
#     else:
#         cap = cv2.VideoCapture(args.video_path)
#         total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # init tracker
#     frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
#     tracker = Track(frame_rate, cfg_file=args.cfg_file, opts=args.opts)  #
#
#     output_path = os.path.join(os.path.abspath(args.output_path), "results")
#     mkdir_p(output_path)
#
#     frame = 0
#     while cap.isOpened():
#         frame += 1
#         if not args.cam:
#             if frame > total_frame:
#                 break
#         _, img = cap.read()
#
#         logging_rank("Processing {} frame ...".format(frame))
#         tracking_res = tracker(img)
#
#         im_dets = tracking_res['im_dets']
#         im_ids  = tracking_res['im_ids']
#
#         # plot results
#         online_im = plot_tracking(img, im_dets, im_ids)
#         save_path = os.path.join(output_path, "{:06d}.jpg".format(frame))
#         cv2.imwrite(save_path, online_im)
def main():
    img = cv2.imread('/home/user/workspace/xxs/DPH_Server/data/test.png')
    tracker = ObjectApi()
    tracker.get_result0(img)


if __name__ == "__main__":
    main()
