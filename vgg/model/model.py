import math
from typing import List, Union

import torch.nn as nn
import torch.nn.functional as F

from vgg.base import BaseModel


class _Vgg(BaseModel):
    def __init__(self, conv_layers, num_classes=100):
        super().__init__()

        self.conv_layers = conv_layers

        self.fc_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = _flatten(x)
        return self.fc_layers(x)


def _flatten(x):
    return x.view(x.size(0), -1)


class Vgg11(_Vgg):
    def __init__(self, num_classes=100):
        conv_layers = _build_layers(_vgg_cfg["A"])
        super().__init__(conv_layers=conv_layers, num_classes=num_classes)


class Vgg13(_Vgg):
    def __init__(self, num_classes=100):
        conv_layers = _build_layers(_vgg_cfg["B"])
        super().__init__(conv_layers=conv_layers, num_classes=num_classes)


class Vgg16(_Vgg):
    def __init__(self, num_classes=100):
        conv_layers = _build_layers(_vgg_cfg["D"])
        super().__init__(conv_layers=conv_layers, num_classes=num_classes)


class Vgg19(_Vgg):
    def __init__(self, num_classes=100):
        conv_layers = _build_layers(_vgg_cfg["E"])
        super().__init__(conv_layers=conv_layers, num_classes=num_classes)


def _build_layers(cfg: List):
    """Build layers based on cgs array."""
    layers = []
    in_channels = 3
    for l in cfg:
        if _is_pooling_layer(l):
            layers.append(_create_pooling_layer())
        else:
            layers.extend(_create_conv_layer(in_channels, l))
            in_channels = l
    return nn.Sequential(*layers)


def _is_pooling_layer(l: Union[int, str]) -> bool:
    return l == "M"


def _create_pooling_layer():
    """Reduce layer area by 50%."""
    return nn.MaxPool2d(stride=2, kernel_size=2)


def _create_conv_layer(in_channels: int, out_channels: int):
    """Return a valid convultion with kernel size 3x3."""
    return [
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        ),
        nn.ReLU(inplace=True),
    ]


_vgg_cfg = {
    # -------------------------------------> 11 weight layers
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # -------------------------------------> 13 weight layers
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # -------------------------------------> 16 weight layers with 1D convolution
    "C": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "1D-256",
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    # -------------------------------------> 16 weight layers
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    # -------------------------------------> 19 weight layers
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CifarMnistModel(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x