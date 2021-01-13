# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_VisualTest import fill_holes
from matplotlib.pyplot import imsave
import cv2


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Dataset/ChineseAsPseudo/ChinaSet_AllFiles/NewPseudo',
                        help='Path to test data')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                    "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                    "----\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (gepengai.ji@gamil.com)\n----\n".format(opt), "#" * 20)


    gt_root = '{}/GT/'.format(opt.data_path)

    print("start fill")

    fill_holes(gt_root)

    print("end fill")


if __name__ == "__main__":
    inference()
