import torch
import torch.nn as nn
from GTN.parameters import *
import torch.nn.functional as F


class Teacher(nn.Module):
    '''
    Implements a Teacher module.
    '''

    def __init__(self):
        super().__init__()

        conv1_filters = 64
        fc1_size = 1024

        self.fc2_filters = 128
        self.fc2_width = img_size
        fc2_size = self.fc2_filters * self.fc2_width * self.fc2_width

        self.fc1 = nn.Linear(noise_size + num_classes, fc1_size)
        nn.init.kaiming_normal_(self.fc1.weight, 0.1)
        self.bn_fc1 = nn.BatchNorm1d(fc1_size, momentum=0.1)

        self.fc2 = nn.Linear(fc1_size, fc2_size)
        nn.init.kaiming_normal_(self.fc2.weight, 0.1)
        self.bn_fc2 = nn.BatchNorm2d(self.fc2_filters, momentum=0.1)

        self.conv1 = nn.Conv2d(self.fc2_filters, conv1_filters, 3, 1, padding=3 // 2)
        self.bn_conv1 = nn.BatchNorm2d(conv1_filters, momentum=0.1)

        self.conv2 = nn.Conv2d(conv1_filters, 1, 3, 1, padding=3 // 2)
        self.bn_conv2 = nn.BatchNorm2d(1, momentum=0.1)

        self.tanh = nn.Tanh()

        self.learner_optim_params = nn.Parameter(torch.tensor([0.02, 0.5]), True)

    def forward(self, x, target):
        '''
        Synthesizes a batch of training examples for the learner.
        Args:
            x (torch.tensor): shape (b, 64)
            target (torch.tensor): shape (b, 10)
        '''
        # Fully connected block 1
        x = torch.cat([x, target], dim=1)  # shape (b, 64+10)
        x = self.fc1(x)  # shape (b, 1024)
        x = F.leaky_relu(x, 0.1)
        x = self.bn_fc1(x)

        # Fully connected block 2
        x = self.fc2(x)  # shape (b, 128*28*28)
        x = F.leaky_relu(x, 0.1)
        x = x.view(  # shape (b, 128, 28, 28)
            -1, self.fc2_filters, self.fc2_width, self.fc2_width
        )
        x = self.bn_fc2(x)

        # Convolutional block 1
        x = self.conv1(x)  # shape (b, 64, 28, 28)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x, 0.1)

        # Convolutional block 2
        x = self.conv2(x)  # shape (b, 1, 28,  28)
        x = self.bn_conv2(x)

        x = (self.tanh(x) + 1 - 2 * mnist_mean) / (2 * mnist_std)
        return x, target
