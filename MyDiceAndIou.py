import torch
from torch.autograd import Variable
import os
import argparse
import numpy as np
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.dataloader_LungInf import dice_test_dataset
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from matplotlib.pyplot import imsave

# Used for Inf-Net and Semi-Inf-Net, and immediate after calculate Dice and IOU and save the result in text file.
# (without holes filling and elimination of extra zones)

def iou(pred,mask):
    inter = np.logical_and(pred,mask)
    union = np.logical_or(pred,mask)
    return inter.sum()/union.sum()

def dice(pred,mask):
    inter = np.logical_and(pred, mask)
    union = np.logical_or(pred,mask)
    return (inter.sum()*2)/(union.sum()+inter.sum())

def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, train_save):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]    # replace your desired scale, try larger scale for better accuracy in small object
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()
            # ---- rescaling the inputs (img/gt/edge) ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(images)
            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = BCE(lateral_edge, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, '
                  'lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

    # ---- save model_lung_infection ----
    save_path = './Snapshots/save_weights/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'Inf-Net-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'Inf-Net-%d.pth' % (epoch+1))

    return model



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=6,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=False,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers in dataloader. In windows, set num_workers=0')
    # model_lung_infection parameters
    parser.add_argument('--net_channel', type=int, default=32,
                        help='internal channel numbers in the Inf-Net, default=32, try larger for better accuracy')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='Res2Net50',
                        help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50')
    # training dataset
    parser.add_argument('--train_path', type=str,
                        default='./Dataset/ChineseAsPseudo/ChinaSet_AllFiles/Pseudo')
    parser.add_argument('--is_semi', type=bool, default=False,
                        help='if True, you will turn on the mode of `Semi-Inf-Net`')
    parser.add_argument('--is_pseudo', type=bool, default=False,
                        help='if True, you will train the model on pseudo-label')
    parser.add_argument('--train_save', type=str, default=None,
                        help='If you use custom save path, please edit `--is_semi=True` and `--is_pseudo=True`')

    # Test
    parser.add_argument('--testsize', type=int, default=352,
                        help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset/TestingSet/LungSegmentation/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./snapshots/save_weights/Semi-Inf-Net/Inf-Net-100.pth',
                        help='Path to weights file. If `semi-sup`, edit it to `Semi-Inf-Net/Semi-Inf-Net-100.pth`')
    parser.add_argument('--save_path', type=str, default='./Results/LungSegmentation/Inf-Net/',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')

    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)
    # - please asign your prefer backbone in opt.
    if opt.backbone == 'Res2Net50':
        print('Backbone loading: Res2Net50')
        from Code.model_lung_infection.InfNet_Res2Net import Inf_Net
    elif opt.backbone == 'ResNet50':
        print('Backbone loading: ResNet50')
        from Code.model_lung_infection.InfNet_ResNet import Inf_Net
    elif opt.backbone == 'VGGNet16':
        print('Backbone loading: VGGNet16')
        from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
    else:
        raise ValueError('Invalid backbone parameters: {}'.format(opt.backbone))
    model = Inf_Net(channel=opt.net_channel, n_class=opt.n_classes).cuda()

    # ---- load pre-trained weights (mode=Semi-Inf-Net) ----
    # - See Sec.2.3 of `README.md` to learn how to generate your own img/pseudo-label from scratch.
    if opt.is_semi and opt.backbone == 'Res2Net50':
        print('Loading weights from weights file trained on pseudo label')
        model.load_state_dict(torch.load('./snapshots/save_weights/Inf-Net_Pseudo/Inf-Net-100.pth'))
    else:
        print('Not loading weights from weights file')

    # weights file save path
    if opt.is_pseudo and (not opt.is_semi):
        train_save = 'Inf-Net_Pseudo'
    elif (not opt.is_pseudo) and opt.is_semi:
        train_save = 'Semi-Inf-Net'
    elif (not opt.is_pseudo) and (not opt.is_semi):
        train_save = 'Inf-Net'
    else:
        print('Use custom save path')
        train_save = opt.train_save

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams
        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = len(train_loader)



    # ---- start !! -----
    print("#"*20, "\nStart Training (Inf-Net-{})\n{}\n".format(opt.backbone, opt), "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, train_save)



    #test part

    if (not opt.is_pseudo):
        test_model = model
        test_model.cuda()
        test_model.eval()

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
            # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)






            gt = gt.numpy()
            gt = gt.reshape(gt.shape[1:])
            gt = gt.astype(bool)
            gt = gt[0, :, :]

            res = np.where(res < np.max(res)/2, True, False)
            res = np.invert(res)
            imsave(opt.save_path + name, res)

            iouC = iou(res, gt)
            diceC = dice(res, gt)

            iouCoef.append(iouC)
            diceCoef.append(diceC)


            result = {'file': str(name), 'iou': iouC, 'dice': diceC}
            metrics.append(result)

        print("IOU mean = ", np.mean(iouCoef), "\nDice maen = ", np.mean(diceCoef))
        print("IOU max = ", np.max(iouCoef), "\nDice max = ", np.max(diceCoef))
        print("IOU min = ", np.min(iouCoef), "\nDice min = ", np.min(diceCoef))

        output_file = open(opt.save_path + 'metrics.txt', 'w', encoding='utf-8')
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

