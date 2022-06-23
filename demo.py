import os
import sys

import shutil
import numpy as np
from subprocess import call

import torch

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.data_augmentation import preprocess_img
from lib.solver import Solver
from lib.voxel import voxel2obj


DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/checkpoint.pth'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def load_demo_images():
    img_h = cfg.CONST.IMG_H
    img_w = cfg.CONST.IMG_W
    
    imgs = []
    
    for i in range(1):
        img = Image.open('imgs/plane0.png')
        img = img.resize((img_h, img_w), Image.ANTIALIAS)
        img = preprocess_img(img, train=False)
        imgs.append([np.array(img).transpose((2, 0, 1)).astype(np.float32)])
    ims_np = np.array(imgs).astype(np.float32)
    return torch.from_numpy(ims_np)


def main():
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'
    demo_imgs = load_demo_images()
    NetClass = load_model('ResidualGRUNet')

    net = NetClass()
    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    solver = Solver(net)
    solver.load(DEFAULT_WEIGHTS)

    voxel_prediction, _ = solver.test_output(demo_imgs)
    voxel_prediction = voxel_prediction.detach().cpu().numpy()

    voxel2obj(pred_file_name, voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)


if __name__ == '__main__':
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
    
