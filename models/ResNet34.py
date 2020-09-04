import torch as t
import torch.nn.functional as F

import sys

sys.path.append('E:/PycharmProjects/DogVSCat/models/BasicModule')

from BasicModule import BasicModule


class ResidualBlock(t.nn.Module):
    def __init__(self, num_input, num_out, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = t.nn.Sequential(
            t.nn.Conv2d(num_input, num_out, 3, stride, 1, bias=False),
            t.nn.BatchNorm2d(num_out),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(num_out, num_out, 3, 1, 1, bias=False),
            t.nn.BatchNorm2d(num_out))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.pre = t.nn.Sequential(t.nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                   t.nn.BatchNorm2d(64),
                                   t.nn.ReLU(inplace=True),
                                   t.nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.fc = t.nn.Linear(512, num_classes)

    def _make_layer(self, num_input, num_out, num_block, stride=1):
        shortcut = t.nn.Sequential(
            t.nn.Conv2d(num_input, num_out, 1, stride, bias=False),
            t.nn.BatchNorm2d(num_out))

        layers = []
        layers.append(ResidualBlock(num_input, num_out, stride, shortcut))
        for i in range(1, num_block):
            layers.append(ResidualBlock(num_out, num_out))

        return t.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)

        return self.fc(x)
