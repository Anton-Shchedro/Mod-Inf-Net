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
from Code.utils.dataloader_VisualTest import test_dataset
from matplotlib.pyplot import imsave
import cv2


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Dataset/ChineseAsPseudo/NewPseudo',
                        help='Path to test data')
    parser.add_argument('--save_path', type=str, default='./Results/LungSegmentation/VisualPseudo/',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    parser.add_argument('--is_edge', type=bool, default=False,
                        help='data_path contains Edge folder')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                    "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                    "----\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (gepengai.ji@gamil.com)\n----\n".format(opt), "#" * 20)


    image_root = '{}/Imgs/'.format(opt.data_path)
    gt_root = '{}/GT/'.format(opt.data_path)

    print("resize images if needed")

    if opt.is_edge:
        edge_root = '{}/Edge/'.format(opt.data_path)
        test_loader = test_dataset(image_root, gt_root, edge_root)
    else:
        test_loader = test_dataset(image_root, gt_root)

    print("resize done")

    os.makedirs(opt.save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, mark, name = test_loader.load_data()

        if not opt.is_edge:
            mark = np.array(mark)
            mark = cv2.Canny(mark, 100, 200, 3)

        mark = np.array(mark)
        res = np.array(image)
        #res = cv2.merge((image,image,image))
        #cv2.cvtColor(image, res, cv2.COLOR_GRAY2RGB)  # creat RGB image from grayscale
        res[mark == 255,:] = [255, 0, 0]  # turn edges to red

        imsave(opt.save_path + name, res)


if __name__ == "__main__":
    inference()
