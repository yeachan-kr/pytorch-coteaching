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


class TextCNN(nn.Module):
    def __init__(self,  vocab: dict, num_class: int, drop_rate: float, pre_weight: np.ndarray = None):
        super(TextCNN, self).__init__()
        self.emb_dim = 300
        self.num_filters = [100, 100, 100]
        self.size_filters = [3, 4, 5]

        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=self.emb_dim,
                                      padding_idx=vocab['<pad>'])

        if pre_weight is not None:
            self.embedding.from_pretrained(pre_weight)

        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n, kernel_size=(k, self.emb_dim))
                                          for n, k in zip(self.num_filters, self.size_filters)])
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight.data)

        num_features = sum(self.num_filters)
        self.drop_out = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_class)
        # nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):

        # 1. Embedding layer
        out = self.embedding(x).unsqueeze(1)

        # 2. 1-D convolution layer
        features = []
        for conv in self.conv_layers:
            h = conv(out).squeeze()
            h = F.relu(h)
            h = F.max_pool1d(input=h, kernel_size=h.size(-1)).squeeze()
            features.append(h)
        features = torch.cat(features, dim=-1)

        # 3. two-layered linear layers (with dropout)
        features = self.fc1(features)
        features = F.relu(features)
        features = self.drop_out(features)
        out = self.fc2(features)
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
