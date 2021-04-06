import numpy as np
import os
import argparse
import cv2

# Used for deleting extra zones from resulting or intermediate GT images.

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Results/LungSegmentation/Inf_Net_Final/Inf_and_Semi_Fill',
                        help='Path to images')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)

    gt_root = '{}/'.format(opt.data_path)
    #gt_root = '{}/GT/'.format(opt.data_path)

    print("start fill")

    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = sorted(gts)
    for gt_path in gts:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

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
        for i,z in enumerate(zones):
            color = (255, 255, 255)
            zone = z['zone']
            c = contours[zone]
            drawing = np.zeros((gt.shape[0], gt.shape[1], 1), dtype=np.uint8)
            drawing = cv2.drawContours(drawing, [c], 0, color, thickness=cv2.FILLED, lineType = 4)
            image = cv2.add(image,drawing)

        inter = np.logical_and(gt, image)
        inter = np.where(inter == True, 255, 0)

        cv2.imwrite(gt_path, inter)

    print("end fill")

if __name__ == "__main__":
    inference()

