import os
from PIL import Image
import cv2
import numpy as np
from skimage.morphology import reconstruction



class test_dataset:
    def __init__(self, image_root, gt_root, edge_root = None):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if (edge_root != None):
            self.edge_flage = True
            self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
            self.edges = sorted(self.edges)
        else:
            self.edge_flage = False

        self.filter_files()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        if self.edge_flage:
            gt = self.binary_loader(self.edges[self.index])
        else:
            gt = self.binary_loader(self.gts[self.index])
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

    def resize(self, img, gt):
        w, h = img.size
        #if h < self.trainsize or w < self.trainsize:
           # h = max(h, self.trainsize)
           # w = max(w, self.trainsize)
           # return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        # else:
        return img, gt.resize((w, h), Image.NEAREST)

    def resizeEdge(self, img, gt, edge):
        w, h = img.size
        #if h < self.trainsize or w < self.trainsize:
           # h = max(h, self.trainsize)
           # w = max(w, self.trainsize)
           # return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        # else:
        return img, gt.resize((w, h), Image.NEAREST), edge.resize((w, h), Image.NEAREST)

    def filter_files(self):
        if self.edge_flage:
            images = []
            gts = []
            edges = []
            for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                ed = Image.open(edge_path)
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
                    edges.append(edge_path)
                else:
                    img, gt, ed = self.resizeEdge(img, gt, ed)
                    gt.save(gt_path)
                    ed.save(edge_path)
                    images.append(img_path)
                    gts.append(gt_path)
                    edges.append(edge_path)
        else:
            assert len(self.images) == len(self.gts)
            images = []
            gts = []
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
            self.images = images
            self.gts = gts


class fill_holes:

    def __init__(self, gt_root):
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = sorted(self.gts)
        self.fill()

    def fill(self):
        gts = []
        for gt_path in self.gts:
            # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE);

            seed = np.copy(gt)
            seed[1:-1, 1:-1] = gt.max()
            mask = gt

            img_fill_holes = reconstruction(seed, mask, method='erosion')

            new_p = Image.fromarray(img_fill_holes)
            new_p = new_p.convert("L")
            new_p.save(gt_path)
            gts.append(gt_path)
        self.gts = gts
