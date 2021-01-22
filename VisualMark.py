import numpy as np
import os
import argparse
from Code.utils.dataloader_VisualTest import test_dataset
from matplotlib.pyplot import imsave
import cv2

# Used to apply edge mask (or if exists only GT, first creates edge mask) on RX image

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Dataset/ChineseAsPseudo/NewPseudo',
                        help='Path to test data')
    parser.add_argument('--save_path', type=str, default='./Results/LungSegmentation/VisualPseudo/',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    parser.add_argument('--is_edge', type=bool, default=False,
                        help='data_path contains Edge folder')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)


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
