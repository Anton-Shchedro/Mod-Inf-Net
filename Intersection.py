import numpy as np
import os
import argparse
import cv2
from matplotlib.pyplot import imsave

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inf_path', type=str, default='./Results/LungSegmentation/Inf_Net_Final/Mont_Fill',
                        help='Path of Inf-Net resulting Image')
    parser.add_argument('--semi_path', type=str, default='./Results/LungSegmentation/NewFillPseudo/MontFill',
                        help='Path of Semi-Inf-Net resulting Image')
    parser.add_argument('--save_path', type=str, default='./Results/LungSegmentation/Inf_Net_Final/Inf_and_Semi/',
                        help='Path to save the predictions.')
    opt = parser.parse_args()

    inf_root = '{}/'.format(opt.inf_path)
    semi_root = '{}/'.format(opt.semi_path)

    infs = [inf_root + f for f in os.listdir(inf_root) if f.endswith('.jpg') or f.endswith('.png')]
    infs = sorted(infs)
    semis = [semi_root + f for f in os.listdir(semi_root) if f.endswith('.jpg') or f.endswith('.png')]
    semis = sorted(semis)

    for inf_path,semi_path in zip(infs,semis):
        inf = cv2.imread(inf_path, cv2.IMREAD_GRAYSCALE)
        semi = cv2.imread(semi_path, cv2.IMREAD_GRAYSCALE)

        inter = np.logical_and(inf, semi)

        name = os.path.basename(inf_path)
        imsave(opt.save_path + name, inter, cmap='gray')


if __name__ == "__main__":
    inference()