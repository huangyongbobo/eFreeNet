import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RSOC_Dataset(Dataset):
    """
    Custom RSOC dataset Class

    Arguments:
        root: Images list
        train: Mode of run model

    Returns:
        img: Tensor of image
        target: Ground truth count
    """
    def __init__(self, root, train=True):
        self.nSamples = len(root)
        self.lines = root
        self.transform = transforms.Compose([transforms.Resize((512, 512)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             ])
        self.train = train

    # The function to get images and ground truth
    def load_data(self, img_path, train):
        img = Image.open(img_path).convert('RGB')
        target = np.load(img_path.replace('.jpg', '.npy').replace('images', 'target_center'))
        if train:
            img = self.data_aug(img)
        target = np.sum(target)

        return img, target

    # Data augmentation
    def data_aug(self, img):
        random_number = random.randint(0, 5)
        if random_number <= 2:
            random_number2 = random.randint(0, 8)
            if random_number2 <= 2:
                img = img.transpose(Image.ROTATE_90)
            elif 3 <= random_number2 <= 5:
                img = img.transpose(Image.ROTATE_180)
            else:
                img = img.transpose(Image.ROTATE_270)
        elif random_number == 3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif random_number == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            img = img
        return img

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img, target = self.load_data(img_path, self.train)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
