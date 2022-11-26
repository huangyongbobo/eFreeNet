import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils, models


class VggNetModel(nn.Module):
    def __init__(self, model):
        super(VggNetModel, self).__init__()
        self.vggNet16_layer = model
        new_classifier = nn.Sequential(*list(self.vggNet16_layer.classifier.children())[:-1])
        self.vggNet16_layer.classifier = new_classifier
        self.layerlist = nn.ModuleList()
        self.layerlist2 = nn.ModuleList()

        self.layer1 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer1)
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer2)
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer3)
        self.layer4 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer4)
        self.layer5 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer5)
        self.layer6 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer6)
        self.layer7 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer7)
        self.layer8 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.layerlist.append(self.layer8)

        self.layer9 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer9)
        self.layer10 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer10)
        self.layer11 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer11)
        self.layer12 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer12)
        self.layer13 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer13)
        self.layer14 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer14)
        self.layer15 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer15)
        self.layer16 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
        )
        self.layerlist2.append(self.layer16)

    def forward(self, x):
        feature = self.vggNet16_layer(x).view(x.size(0), -1)
        for index in range(8):
            if index == 0:
                pred = self.layerlist[index](feature[:, index * 512:(index+1) * 512])
                output1 = pred.transpose(0, 1)
                output2 = self.layerlist2[index](pred).transpose(0, 1)
            else:
                pred = self.layerlist[index](feature[:, index * 512:(index + 1) * 512])
                output1 = torch.cat((output1, pred.transpose(0, 1)),)
                output2 = torch.cat((output2, self.layerlist2[index](pred).transpose(0, 1)),)
        return output1, output2
