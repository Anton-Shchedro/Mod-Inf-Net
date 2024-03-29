# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class COVIDDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if len(edge_root) != 0:
            self.edge_flage = True
            self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
            self.edges = sorted(self.edges)
        else:
            self.edge_flage = False

        self.filter_files() # modifyed this function
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt

    '''
    originally:
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts
        
        
    in modified version when img.size == gt.size False
    gt is resized up to size of img
    in case when edge exist, edge is also resized
    '''

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        if not self.edge_flage:
            for img_path, gt_path in zip(self.images, self.gts):
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
                else:
                    img, gt = self.resize(img,gt)
                    gt.save(gt_path)
                    images.append(img_path)
                    gts.append(gt_path)
        else:
            edges = []
            for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                edge = Image.open(edge_path)
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
                    edges.append(edge_path)
                else:
                    img, gt, edge = self.resizeEdge(img,gt,edge)
                    gt.save(gt_path)
                    edge.save(edge_path)
                    images.append(img_path)
                    gts.append(gt_path)
                    edges.append(edge_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        w, h = img.size
        # train size is most times significantly lower then img size, so following code doesn't necessary.
        #if h < self.trainsize or w < self.trainsize:
           # h = max(h, self.trainsize)
           # w = max(w, self.trainsize)
           # return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        # else:
        return img, gt.resize((w, h), Image.NEAREST)

    def resizeEdge(self, img, gt, edge):
        # little modification of resize function in case when edge exists.
        w, h = img.size
        #if h < self.trainsize or w < self.trainsize:
           # h = max(h, self.trainsize)
           # w = max(w, self.trainsize)
           # return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        # else:
        return img, gt.resize((w, h), Image.NEAREST), edge.resize((w, h), Image.NEAREST)

    def __len__(self):
        return self.size


class LoadCVDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize,k_size,k):
        self.trainsize = trainsize

        self.k_size = k_size
        self.k = k



        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        start = int((len(self.images) / self.k_size) * self.k)
        end = int(((len(self.images) / self.k_size) * (self.k + 1)) - 1)

        self.images = sorted(self.images)

        self.gts = sorted(self.gts)



        self.test_images = self.images[start:end]

        self.images = [x for x in self.images if x not in self.test_images]

        self.images = sorted(self.images)
        self.test_images = sorted(self.test_images)



        self.test_gts = self.gts[start:end]

        self.gts = [x for x in self.gts if x not in self.test_gts]

        self.gts = sorted(self.gts)
        self.test_gts = sorted(self.test_gts)



        if len(edge_root) != 0:
            self.edge_flage = True
            self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
            self.edges = sorted(self.edges)

            self.test_edges = self.edges[start:end]

            self.edges = [x for x in self.edges if x not in self.test_edges]

            self.edges = sorted(self.edges)
            self.test_edges = sorted(self.test_edges)
        else:
            self.edge_flage = False

        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = COVIDDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader

def get_cv_loader(image_root, gt_root, edge_root,k_size,k,testsize, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = LoadCVDataset(image_root, gt_root, edge_root, trainsize,k_size,k)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    test_loader = cv_test_dataset(dataset.test_images,dataset.test_gts,trainsize)
    return test_loader, data_loader



class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        image = self.transform(image).unsqueeze(0)
        # gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

class dice_test_dataset:
    # used in MyDiceAndIOU and MyTDice
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.bin_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.bin_transform(gt).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

class cv_test_dataset:
    def __init__(self, images, gts, testsize):
        self.testsize = testsize
        self.images = images
        self.gts = gts
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        #self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.bin_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.bin_transform(gt).unsqueeze(0)
        # gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, name, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')