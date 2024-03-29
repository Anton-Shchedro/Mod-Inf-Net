import torch
import numpy as np
import os
import argparse
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import dice_test_dataset
from matplotlib.pyplot import imsave

# Used as MyTBin, but immediate after calculate Dice and IOU and save the result in text file.

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
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset/TrainingSet/LungSegmentation/Montgomery',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./snapshots/save_weights/Inf-Net/Inf-Net-100.pth',
                        help='Path to weights file. If `semi-sup`, edit it to `Semi-Inf-Net/Semi-Inf-Net-100.pth`')
    parser.add_argument('--save_path', type=str, default='./Results/LungSegmentation/IOUandDICE',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)

    model = Network()
    model.load_state_dict(torch.load(opt.pth_path, map_location={'cuda:1':'cuda:0'}))
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(opt.data_path)
    gt_root = '{}/GT/'.format(opt.data_path)
    test_loader = dice_test_dataset(image_root, gt_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    iouCoef = []
    diceCoef = []

    metrics = []

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

        res = lateral_map_2
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)



        # change Tensor image to numpy image
        gt = gt.numpy()
        gt = gt.reshape(gt.shape[1:])
        gt = gt.astype(bool)
        gt = gt[0, :, :]

        res = np.where(res < np.max(res)/2, True, False)
        res = np.invert(res)
        imsave(opt.save_path + name, res, cmap = 'gray')

        iouC = iou(res, gt)
        diceC = dice(res, gt)

        iouCoef.append(iouC)
        diceCoef.append(diceC)


        result = {'file': str(name), 'iouC': iouC, 'dice': diceC}
        metrics.append(result)

    print("IOU mean = ", np.mean(iouCoef), "\nDice maen = ", np.mean(diceCoef))
    print("IOU max = ", np.max(iouCoef), "\nDice max = ", np.max(diceCoef))
    print("IOU min = ", np.min(iouCoef), "\nDice min = ", np.min(diceCoef))

    output_file = open(opt.save_path + 'metrics.txt', 'w', encoding='utf-8')
    output_file.write("Name, IOU, Dice\n")

    for item in metrics:
        for d in item.items():
            output_file.write(str(d)+ "    ")
        output_file.write("\n")
    output_file.write("\nIOU mean = " + str(np.mean(iouCoef)))
    output_file.write("\nDice mean = " + str(np.mean(diceCoef)))
    output_file.write("\nIOU max = " + str(np.max(iouCoef)))
    output_file.write("\nDice max = " + str(np.max(diceCoef)))
    output_file.write("\nIOU min = " + str(np.min(iouCoef)))
    output_file.write("\nDice min = " + str(np.min(diceCoef)))


if __name__ == "__main__":
    inference()
