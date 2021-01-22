import torch
import numpy as np
import os
import argparse
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset
from matplotlib.pyplot import imsave

# Used as MyTest..., but with threshold and only Res2Net as backbone

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset/TestingSet/LungSegmentation/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./Snapshots/save_weights/Inf-Net/Inf-Net-100.pth',
                        help='Path to weights file. If `semi-sup`, edit it to `Semi-Inf-Net/Semi-Inf-Net-100.pth`')
    parser.add_argument('--save_path', type=str, default='./Results/LungSegmentation/Inf-Net/',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\n".format(opt), "#" * 20)

    model = Network()
    model.load_state_dict(torch.load(opt.pth_path, map_location={'cuda:1':'cuda:0'}))
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()

        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

        res = lateral_map_2
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = np.where(res < np.max(res) / 2, True, False)
        res = np.invert(res)
        imsave(opt.save_path + name, res, cmap = 'gray')

    print('Test Done!')


if __name__ == "__main__":
    inference()
