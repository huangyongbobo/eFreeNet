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
from Dataset.RSOC_dataset import RSOC_Dataset
from Dataset.VisDronePeople_dataset import VD_people_Dataset
from Dataset.VisDroneVehicle_dataset import VD_vehicle_Dataset
from model import eFreeNet
from ranking_loss import Ranking_loss
from ambiguity_loss import Ambiguity_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train of eFreeNet')

    parser.add_argument('--dataset', type=str, default='RSOC_building', help='object counting dataset',
                        choices=['RSOC_building', 'VisDronePeople', 'VisDroneVehicle'])
    parser.add_argument('--train_dir', type=str,
                        default='/media/ysliu/6b94d4ca-f5c4-46ae-8497-af46d2544dfc/Maoer'
                                '/RSOC_building/train_data/images', help='training set path')
    parser.add_argument('--val_dir', type=str,
                        default='/media/ysliu/6b94d4ca-f5c4-46ae-8497-af46d2544dfc/Maoer'
                                '/RSOC_building/val_data/images', help='validate set path')
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
    # RSOC_building：train_max-142, train_min-17
    # VisDronePeople: train_max-289, train_min-10
    # VisDroneVehicle: train_max-349,train_min-10
    parser.add_argument('--T', type=float, default=30, help='relevance coefficient in ranking_loss')
    parser.add_argument('--lambdas', type=float, default=0.1, help='weight of ambiguity_loss')
    parser.add_argument('--miu', type=float, default=0.1, help='weight of estimation loss')

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
    model = eFreeNet(vggNet16).to(device)

    loss_function = Ranking_loss().to(device)
    loss_function2 = Ambiguity_loss().to(device)

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
            torch.save(model.state_dict(), './eFreeNet.pth')


def train(args, model, loss_function, loss_function2, optimizer, epoch, device):
    if args.dataset == 'RSOC_building':
        train_list = glob.glob(os.path.join(args.train_dir, '*.jpg'))
        train_dataset = RSOC_Dataset(train_list, train=True)
    elif args.dataset == 'VisDronePeople':
        ground_path = args.train_dir.replace('images', 'Ground_Truth')
        train_dataset = VD_people_Dataset(args.train_dir, ground_path, train=True)
    elif args.dataset == 'VisDroneVehicle':
        ground_path = args.train_dir.replace('images', 'Ground_Truth')
        train_dataset = VD_vehicle_Dataset(args.train_dir, ground_path, train=True)
    else:
        raise ValueError('Non-existent train dataset')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers)
    train_bar = tqdm(train_loader)
    running_loss = 0.0
    ranking_loss_num = 0.0
    ambiguity_loss_num = 0.0
    estimation_loss_num = 0.0
    for step, data in enumerate(train_bar):
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        images, labels = images.to(device), labels.to(device)
        labels = labels.float()
        MaxValue, Max_index = torch.max(labels, dim=0)
        MinValue, Min_index = torch.min(labels, dim=0)
        output1, output2 = model(images)
        optimizer.zero_grad()
        ranking_loss = 0
        estimation_loss = 0
        for index in range(args.ensemble_num):
            pred = output1[index, :]
            ranking_loss += loss_function(pred, labels, Max_index, Min_index, args.max_target, args.min_target, args.T)
            pred2 = output2[index, :]
            estimation_loss += torch.sum(((pred2 - labels) ** 2) * 1 / 2) / len(labels)

        pred_avg = torch.sum(output2, dim=0)
        pred_avg = pred_avg / args.ensemble_num
        ambiguity_loss = loss_function2(output2.to(device), pred_avg.to(device))

        loss = ranking_loss.to(device) + args.lambdas * ambiguity_loss + args.miu * estimation_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        ranking_loss_num += ranking_loss.item()
        ambiguity_loss_num += ambiguity_loss.item()
        estimation_loss_num += estimation_loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.5f} ranking:{:.5f} " \
                         "ambiguity:{:.5f} estimation:{:.5f}".format(epoch + 1, args.max_epoch,
                                                                     running_loss / len(train_loader),
                                                                     ranking_loss_num / len(train_loader),
                                                                     ambiguity_loss_num / len(train_loader),
                                                                     estimation_loss_num / len(train_loader))


def validate(args, model, epoch, device):
    model.eval()
    if args.dataset == 'RSOC_building':
        val_list = glob.glob(os.path.join(args.val_dir, '*.jpg'))
        val_dataset = RSOC_Dataset(val_list, train=False)
    elif args.dataset == 'VisDronePeople':
        ground_path = args.val_dir.replace('images', 'Ground_Truth')
        val_dataset = VD_people_Dataset(args.val_dir, ground_path, train=False)
    elif args.dataset == 'VisDroneVehicle':
        ground_path = args.val_dir.replace('images', 'Ground_Truth')
        val_dataset = VD_vehicle_Dataset(args.val_dir, ground_path, train=False)
    else:
        raise ValueError('Non-existent validate dataset')
    validate_loader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                 num_workers=args.num_workers)
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
