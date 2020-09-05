""" Modeling layer Implementation """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def call_bn(bn, x):
    return bn(x)


class NoiseModel(nn.Module):
    def __init__(self, num_class: int):
        super(NoiseModel, self).__init__()
        self.num_class = num_class
        self.transition_mat = Parameter(torch.eye(num_class))

    def forward(self, x):
        """
        x:
            shape = (batch, num_class) (probability distribution)
        return:
            noise distribution
        """
        out = torch.matmul(x, self.transition_mat)
        return out


class CNN(nn.Module):
    def __init__(self, input_channel=3, num_class=10, dropout_rate=0.25, top_bn=False):
        super(CNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn

        # CNN
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(128, num_class)

        # Batch norm
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.fc(h)
        return logit
