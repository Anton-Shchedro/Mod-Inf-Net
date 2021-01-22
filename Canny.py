import os
import argparse
import cv2

# used to make edge when creating GT image and need work with edges of that GT

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Dataset/ChineseAsPseudo/ChinaSet_AllFiles/NewPseudo',
                        help='Path to test data')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)


    gt_root = '{}/GT/'.format(opt.data_path)
    edge_root = '{}/Edge/'.format(opt.data_path)

    print("start canny")

    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = sorted(gts)

    os.makedirs(edge_root, exist_ok=True)

    for gt_path in gts:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        canny_output = cv2.Canny(gt, 100, 200,3)

        #image = image[:,:,-1]
        name = gt_path.replace('/GT/','/Edge/')
        cv2.imwrite(name, canny_output)


    print("end canny")


if __name__ == "__main__":
    inference()

