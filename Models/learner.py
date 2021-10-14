import numpy as np
import torch
import torch.nn as nn
from GTN.parameters import *
import torch.nn.functional as F


class Learner(nn.Module):
    '''
    Implements a Learner module.
    '''

    def __init__(self, num_conv1=None, num_conv2=None):
        super().__init__()

        # Randomly select and evaluate convolutional depth
        # for evaluation/comparison in neural architecture search
        if num_conv1 is None:
            conv1_filters = np.random.randint(32, 64)
        else:
            conv1_filters = num_conv1
        if num_conv2 is None:
            conv2_filters = np.random.randint(64, 128)
        else:
            conv2_filters = num_conv2

        self.conv1 = nn.Conv2d(1, conv1_filters, 3, 1)
        self.bn1 = nn.BatchNorm2d(conv1_filters, momentum=0.1)

        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, 1)
        self.bn2 = nn.BatchNorm2d(conv2_filters, momentum=0.1)

        c1_size = (img_size - 3 + 1) // 2
        c2_size = (c1_size - 3 + 1) // 2

        self.fc = nn.Linear(conv2_filters * c2_size * c2_size, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes, momentum=0.1)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn3(x)

        return x
