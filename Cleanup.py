import argparse
from skimage.morphology import reconstruction
import cv2
import numpy as np
import os
from PIL import Image

# Used for holes filling of resulting or intermediate GT images.

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Results/LungSegmentation/NewPseudo/MontFill',
                        help='Path to test data')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)

    gt_root = '{}/'.format(opt.data_path)
    #gt_root = '{}/GT/'.format(opt.data_path)

    print("start fill")

    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = sorted(gts)
    for gt_path in gts:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE);

        seed = np.copy(gt)
        seed[1:-1, 1:-1] = gt.max()
        mask = gt

        img_fill_holes = reconstruction(seed, mask, method='erosion')

        new_p = Image.fromarray(img_fill_holes)
        new_p = new_p.convert("L")
        new_p = np.array(new_p)

        contours, hierarchy = cv2.findContours(new_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        zones = []
        if len(contours) > 2:
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                z = {'area': area, 'zone': i}
                zones.append(z)
            zones = sorted(zones, key=lambda i: i['area'], reverse=True)
            zones = zones[:2]
        else:
            for i, c in enumerate(contours):
                z = {'area': 0, 'zone': i}
                zones.append(z)
        image = np.zeros((new_p.shape[0], new_p.shape[1], 1), dtype=np.uint8)
        for i, z in enumerate(zones):
            color = (255, 255, 255)
            zone = z['zone']
            c = contours[zone]
            drawing = np.zeros((new_p.shape[0], new_p.shape[1], 1), dtype=np.uint8)
            drawing = cv2.drawContours(drawing, [c], 0, color, thickness=cv2.FILLED, lineType=4)
            image = cv2.add(image, drawing)

        cv2.imwrite(gt_path, image)
    print("end fill")


if __name__ == "__main__":
    inference()
