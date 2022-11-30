import os
import time
import pandas as pd
import torch
import glob
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, utils, models
from torchvision import transforms
from tqdm import tqdm
from Dataset.RSOC_dataset import RSOC_Dataset
from Dataset.VisDronePeople_dataset import VD_people_Dataset
from Dataset.VisDroneVehicle_dataset import VD_vehicle_Dataset
from model import eFreeNet


def parse_args():
    parser = argparse.ArgumentParser(description='Test of eFreeNet')

    parser.add_argument('--dataset', type=str, default='RSOC_building', help='object counting dataset',
                        choices=['RSOC_building', 'VisDronePeople', 'VisDroneVehicle'])
    parser.add_argument('--test_dir', type=str, default='.../RSOC_building/test_data/images', help='test set path')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch_size of test')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ensemble_num', type=int, default=8, help='number of members of ensemble learning')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if args.dataset == 'RSOC_building':
        test_list = glob.glob(os.path.join(args.test_dir, '*.jpg'))
        test_dataset = RSOC_Dataset(test_list, train=False)
    elif args.dataset == 'VisDronePeople':
        ground_path = args.test_dir.replace('images', 'Ground_Truth')
        test_dataset = VD_people_Dataset(args.test_dir, ground_path, train=False)
    elif args.dataset == 'VisDroneVehicle':
        ground_path = args.test_dir.replace('images', 'Ground_Truth')
        test_dataset = VD_vehicle_Dataset(args.test_dir, ground_path, train=False)
    else:
        raise ValueError('Non-existent validate dataset')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    vggNet16 = models.vgg16_bn(pretrained=True)
    model = eFreeNet(vggNet16)

    model = model.to(device)

    assert os.path.exists('./eFreeNet.pth'), "file: '{}' dose not exist.".format('./eFreeNet.pth')
    model.load_state_dict(torch.load('./eFreeNet.pth'))

    model.eval()

    AE = 0
    SE = 0
    test_bar = tqdm(test_loader)
    for step, data in enumerate(test_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output1, output2 = model(images)
            output2 = torch.squeeze(output2, dim=1)
        pre_count = torch.sum(output2) / args.ensemble_num
        AE += abs(pre_count - labels)
        SE += (pre_count - labels) * (pre_count - labels)

    AE = AE.cpu().numpy()
    SE = SE.cpu().numpy()

    MAE = AE / len(test_dataset)
    RMSE = np.sqrt(SE / len(test_dataset))

    print('MAE: %.5f' % MAE)
    print('RMSE: %.5f' % RMSE)
