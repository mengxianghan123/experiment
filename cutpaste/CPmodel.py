import torch
import torch.nn as nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, arch, aug_mode, pretrained):
        super(Model, self).__init__()

        self.resnet = models.__dict__[arch](pretrained=pretrained)

        self.num_cls = 2 if aug_mode == "cut_paste" or aug_mode == "scar" else 3

        head1_list = []
        head1_list.append(nn.Linear(self.resnet.fc.weight.shape[1], 128))
        head1_list.append(nn.BatchNorm1d(128))
        head1_list.append(nn.ReLU(inplace=True))
        head1_list.append(nn.Linear(128, self.num_cls))
        # head1_list.append(nn.BatchNorm1d(64))
        # head1_list.append(nn.ReLU(inplace=True))
        # head1_list.append(nn.Linear(64, self.num_cls))
        # head1_list.append(nn.Linear(self.resnet.fc.weight.shape[1], self.num_cls))
        self.head1 = nn.Sequential(*head1_list)

        # head2_list = []
        # head2_list.append(nn.Linear(self.resnet.fc.weight.shape[1], 128))
        # head2_list.append(nn.BatchNorm1d(128))
        # head2_list.append(nn.ReLU(inplace=True))
        # head2_list.append(nn.Linear(128, 64))
        # head2_list.append(nn.BatchNorm1d(64))
        # head2_list.append(nn.ReLU(inplace=True))
        # head2_list.append(nn.Linear(64, 2))
        # head2_list.append(nn.Linear(self.resnet.fc.weight.shape[1], 2))
        # self.head2 = nn.Sequential(*head2_list)

        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.resnet.flatten = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        # self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        # self.conv1 = nn.Conv2d(512, 128, 3, 1, 1)
        # self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        # self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        # self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        # self.conv5 = nn.Conv2d(16, 3, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.bn4 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.resnet(x)
        embed = torch.flatten(self.avgpool(y), 1)
        paste = self.head1(embed)
        # disorder = self.head2(embed)

        # y = self.upsample2(y)
        # y = self.conv1(y)
        # y = self.bn1(y)
        # y = self.relu(y)

        # y = self.upsample2(y)
        # y = self.conv2(y)
        # y = self.bn2(y)
        # y = self.relu(y)

        # y = self.upsample2(y)
        # y = self.conv3(y)
        # y = self.bn3(y)
        # y = self.relu(y)

        # y = self.upsample2(y)
        # y = self.conv4(y)
        # y = self.bn4(y)
        # y = self.relu(y)

        # y = self.upsample2(y)
        # y = self.conv5(y)

        # return embed, paste, disorder, y
        return embed, paste
