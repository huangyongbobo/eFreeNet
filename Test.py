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
from Dataset.RSOC_Dataset import listDataset
from Dataset.VisDronePeople_Dataset import cdpeopleDataset
from Dataset.VisDroneVehicle_Dataset import cdvehicleDataset
from model import VggNetModel


def parse_args():
    parser = argparse.ArgumentParser(description='Test of eFateNet')

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
        test_transform = transforms.Compose([transforms.Resize((512, 512)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             ])
        test_list = glob.glob(os.path.join(args.test_dir, '*.jpg'))
        test_dataset = listDataset(test_list, transform=test_transform, train=False, num_workers=args.num_workers)
    elif args.dataset == 'VisDronePeople':
        test_transform = transforms.Compose([transforms.Resize((512, 768)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             ])
        ground_path = args.test_dir.replace('images', 'Ground_Truth')
        test_dataset = cdpeopleDataset(args.test_dir, ground_path, test_transform, train=False)
    elif args.dataset == 'VisDroneVehicle':
        test_transform = transforms.Compose([transforms.Resize((512, 768)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             ])
        ground_path = args.test_dir.replace('images', 'Ground_Truth')
        test_dataset = cdvehicleDataset(args.test_dir, ground_path, test_transform, train=False)
    else:
        raise ValueError('Non-existent validate dataset')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)

    vggNet16 = models.vgg16_bn(pretrained=True)
    model = VggNetModel(vggNet16)

    model = model.to(device)

    assert os.path.exists('./eFateNet.pth'), "file: '{}' dose not exist.".format('./eFateNet.pth')  # 判断是否导入成功
    model.load_state_dict(torch.load('./eFateNet.pth'))

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
