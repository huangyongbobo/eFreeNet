import numpy as np
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import random
import cv2
from torchvision import transforms
from PIL import Image


class cdpeopleDataset(data.Dataset):
    def __init__(self, img_path, gt_path, train):
        self.img_path = img_path
        self.gt_path = gt_path
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.num_samples = len(self.data_files)
        self.transform = transforms.Compose([transforms.Resize((512, 768)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                             ])
        self.train = train

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)

        img = self.transform(img)
        den = np.sum(den)

        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname)).convert('RGB')
        if self.train:
            img = self.data_aug(img)
        den = np.load(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.npy'))
        den = np.squeeze(den)

        return img, den

    def data_aug(self, img):
        random_number = random.randint(0, 1)
        if random_number == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img
        return img

    def get_num_samples(self):
        return self.num_samples
