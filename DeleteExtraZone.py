# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import numpy as np
import os
import argparse
from PIL import Image
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

    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = sorted(gts)
    for gt_path in gts:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        canny_output = cv2.Canny(gt, 100, 200,3)
        contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        zones = []
        if len(contours) > 2:
            for i,c in enumerate(contours):
                area = cv2.contourArea(c)
                z = {'area': area, 'zone': i}
                zones.append(z)
            zones = sorted(zones, key = lambda i: i['area'],reverse=True)
            zones = zones[:2]
        else:
            for i,c in enumerate(contours):
                z = {'area': 0, 'zone': i}
                zones.append(z)
        image = np.zeros((gt.shape[0], gt.shape[1], 1), dtype=np.uint8)
        #image = image[:, :, -1]
        #image = np.where(image == 0, False, True)
        for i,z in enumerate(zones):
            color = (255, 255, 255)
            zone = z['zone']
            c = contours[zone]
            drawing = np.zeros((gt.shape[0], gt.shape[1], 1), dtype=np.uint8)
            drawing = cv2.drawContours(drawing, [c], 0, color, thickness=cv2.FILLED, lineType = 4)
            #drawing = drawing[:, :, -1]
            #drawing = np.where(drawing == 0, False, True)
            image = cv2.add(image,drawing)


        #image = image[:,:,-1]
        cv2.imwrite(gt_path, image)


    print("end fill")


if __name__ == "__main__":
    inference()

