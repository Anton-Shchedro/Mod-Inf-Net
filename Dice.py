import numpy as np
import os
import argparse
import cv2


def iou(pred,mask):
    inter = np.logical_and(pred,mask)
    union = np.logical_or(pred,mask)
    return inter.sum()/union.sum()

def dice(pred,mask):
    inter = np.logical_and(pred, mask)
    union = np.logical_or(pred,mask)
    return (inter.sum()*2)/(union.sum()+inter.sum())

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Results/LungSegmentation/NewFillPseudo/MontFill',
                        help='Path to test data')
    parser.add_argument('--referance', type=str, default='./Dataset/ChineseAsPseudo/Montgomery/Test',
                        help='Path to test data')
    opt = parser.parse_args()
    gt_root = '{}/'.format(opt.data_path)
    ref_root = '{}/GT/'.format(opt.referance)

    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = sorted(gts)
    refs = [ref_root + f for f in os.listdir(ref_root) if f.endswith('.jpg') or f.endswith('.png')]
    refs = sorted(refs)

    iouCoef = []
    diceCoef = []

    metrics = []

    for gt_path,ref_path in zip(gts,refs):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

        ref = cv2.resize(ref, dsize=(352, 352), interpolation=cv2.INTER_LINEAR)

        iouC = iou(ref, gt)
        diceC = dice(ref, gt)

        iouCoef.append(iouC)
        diceCoef.append(diceC)

        name = os.path.basename(gt_path)

        result = {'file': str(name), 'iouC': iouC, 'dice': diceC}
        metrics.append(result)

    print("IOU mean = ", np.mean(iouCoef), "\nDice maen = ", np.mean(diceCoef))
    print("IOU max = ", np.max(iouCoef), "\nDice max = ", np.max(diceCoef))
    print("IOU min = ", np.min(iouCoef), "\nDice min = ", np.min(diceCoef))

    output_file = open(opt.data_path + 'metrics.txt', 'w', encoding='utf-8')
    output_file.write("Name, IOU, Dice\n")

    for item in metrics:
        for d in item.items():
            output_file.write(str(d) + "    ")
        output_file.write("\n")
    output_file.write("\nIOU mean = " + str(np.mean(iouCoef)))
    output_file.write("\nDice mean = " + str(np.mean(diceCoef)))
    output_file.write("\nIOU max = " + str(np.max(iouCoef)))
    output_file.write("\nDice max = " + str(np.max(diceCoef)))
    output_file.write("\nIOU min = " + str(np.min(iouCoef)))
    output_file.write("\nDice min = " + str(np.min(diceCoef)))


if __name__ == "__main__":
    inference()