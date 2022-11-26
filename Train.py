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
from tqdm import tqdm
from Dataset.RSOC_Dataset import listDataset
from Dataset.VisDronePeople_Dataset import cdpeopleDataset
from Dataset.VisDroneVehicle_Dataset import cdvehicleDataset
from model import VggNetModel
from ApproxNDCGLoss import ApproxNDCG
from Neg_cor_loss import Negative_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train of eFreeNet')

    parser.add_argument('--dataset', type=str, default='RSOC_building', help='object counting dataset',
                        choices=['RSOC_building', 'VisDronePeople', 'VisDroneVehicle'])
    parser.add_argument('--train_dir', type=str, default='.../RSOC_building/train_data/images', help='training set path')
    parser.add_argument('--val_dir', type=str, default='.../RSOC_building/val_data/images', help='validate set path')
    parser.add_argument('--max_epoch', type=int, default=400, help='max training epoch')
    parser.add_argument('--train_batch_size', type=int, default=8, help='batch_size of train')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch_size of validate')
    parser.add_argument('--lr1', type=float, default=1 * 1e-5, help='the initial learning rate of CNN backbone ')
    parser.add_argument('--lr2', type=float, default=0.01, help='the initial learning rate of ensemble members')
    parser.add_argument('--weight_decay', type=float, default=5 * 1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3050, help='Random seed')
    parser.add_argument('--ensemble_num', type=int, default=8, help='number of members of ensemble learning')
    parser.add_argument('--max_target', type=int, default=349, help='maximum target in training set')
    parser.add_argument('--min_target', type=int, default=10, help='minimum target in training set')
    parser.add_argument('--T', type=float, default=30, help='hyperparameter in ApproxNDCGLoss')
    parser.add_argument('--lambdas', type=float, default=0.005, help='weight of negative correlation learning loss')
    parser.add_argument('--miu', type=float, default=0.1, help='weight of counting loss')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    vggNet16 = models.vgg16_bn(pretrained=True)
    model = VggNetModel(vggNet16).to(device)

    loss_function = ApproxNDCG().to(device)
    loss_function2 = Negative_loss().to(device)

    member_para_list = []
    for index1 in range(args.ensemble_num):
        ignored_params = list(map(id, model.layerlist2[index1].parameters()))
        member_para_list.append(ignored_params)

    for index2 in range(args.ensemble_num):
        if index2 == 0:
            base_params = list(filter(lambda p: id(p) not in member_para_list[index2], model.parameters()))
        else:
            base_params = list(filter(lambda p: id(p) not in member_para_list[index2], base_params))

    optimizer = optim.SGD([{'params': base_params, 'lr': args.lr1},
                           {'params': model.layer9.parameters(), 'lr': args.lr2},
                           {'params': model.layer10.parameters(), 'lr': args.lr2},
                           {'params': model.layer11.parameters(), 'lr': args.lr2},
                           {'params': model.layer12.parameters(), 'lr': args.lr2},
                           {'params': model.layer13.parameters(), 'lr': args.lr2},
                           {'params': model.layer14.parameters(), 'lr': args.lr2},
                           {'params': model.layer15.parameters(), 'lr': args.lr2},
                           {'params': model.layer16.parameters(), 'lr': args.lr2}
                           ], momentum=args.momentum, weight_decay=args.weight_decay)

    min_mae = float('inf')
    for epoch in range(args.max_epoch):
        model.train()
        adjust_learning_rate(optimizer, epoch)
        train(args, model, loss_function, loss_function2, optimizer, epoch, device)
        MAE = validate(args, model, epoch, device)
        if MAE <= min_mae:
            min_mae = MAE
            torch.save(model.state_dict(), './eFateNet.pth')


def train(args, model, loss_function, loss_function2, optimizer, epoch, device):
    if args.dataset == 'RSOC_building':
        train_list = glob.glob(os.path.join(args.train_dir, '*.jpg'))
        train_dataset = listDataset(train_list, train=True, num_workers=args.num_workers)
    elif args.dataset == 'VisDronePeople':
        ground_path = args.train_dir.replace('images', 'Ground_Truth')
        train_dataset = cdpeopleDataset(args.train_dir, ground_path, train=True)
    elif args.dataset == 'VisDroneVehicle':
        ground_path = args.train_dir.replace('images', 'Ground_Truth')
        train_dataset = cdvehicleDataset(args.train_dir, ground_path, train=True)
    else:
        raise ValueError('Non-existent train dataset')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_bar = tqdm(train_loader)
    running_loss = 0.0
    loss1_num = 0.0
    loss2_num = 0.0
    loss3_num = 0.0
    for step, data in enumerate(train_bar):
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        images, labels = images.to(device), labels.to(device)
        labels = labels.float()
        MaxValue, Max_index = torch.max(labels, dim=0)
        MinValue, Min_index = torch.min(labels, dim=0)
        output1, output2 = model(images)
        optimizer.zero_grad()
        loss1 = 0
        loss3 = 0
        for index in range(args.ensemble_num):
            pred = output1[index, :]
            loss1 += loss_function(pred, labels, Max_index, Min_index, args.max_target, args.min_target, args.T)
            pred2 = output2[index, :]
            loss3 += torch.sum(((pred2 - labels) ** 2) * 1 / 2) / len(labels)

        pred_avg = torch.sum(output2, dim=0)
        pred_avg = pred_avg / args.ensemble_num
        loss2 = loss_function2(output2.to(device), pred_avg.to(device))

        loss1 = loss1.to(device)
        loss = loss1 + args.lambdas * loss2 + args.miu * loss3
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss1_num += loss1.item()
        loss2_num += loss2.item()
        loss3_num += loss3.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.5f} loss1:{:.5f} " \
                         "loss2:{:.5f} loss3:{:.5f}".format(epoch + 1, args.max_epoch,
                                                            running_loss / len(train_loader),
                                                            loss1_num / len(train_loader),
                                                            loss2_num / len(train_loader),
                                                            loss3_num / len(train_loader))


def validate(args, model, epoch, device):
    model.eval()
    if args.dataset == 'RSOC_building':
        val_list = glob.glob(os.path.join(args.val_dir, '*.jpg'))
        val_dataset = listDataset(val_list, train=False, num_workers=args.num_workers)
    elif args.dataset == 'VisDronePeople':
        ground_path = args.val_dir.replace('images', 'Ground_Truth')
        val_dataset = cdpeopleDataset(args.val_dir, ground_path, train=False)
    elif args.dataset == 'VisDroneVehicle':
        ground_path = args.val_dir.replace('images', 'Ground_Truth')
        val_dataset = cdvehicleDataset(args.val_dir, ground_path, train=False)
    else:
        raise ValueError('Non-existent validate dataset')
    validate_loader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False)
    validate_bar = tqdm(validate_loader)
    AE = 0
    for batch_i, (img, target) in enumerate(validate_bar):
        img, target = img.to(device), target.to(device)
        img, target = Variable(img), Variable(target)
        img, target = img.to(device), target.to(device)
        target = target.float()
        with torch.no_grad():
            output1, output2 = model(img)
            output2 = torch.squeeze(output2, dim=1)
        pre_count = torch.sum(output2) / args.ensemble_num
        AE += abs(pre_count - target)

    MAE = AE / len(val_dataset)
    print('[epoch %d] MAE: %.3f' % (epoch + 1, MAE))
    return MAE


def adjust_learning_rate(optimizer, epoch):
    if epoch + 1 == 50:
        for params in optimizer.param_groups:
            params['lr'] *= 0.1
    if epoch + 1 == 100:
        for params in optimizer.param_groups:
            params['lr'] *= 0.5


if __name__ == '__main__':
    main()
