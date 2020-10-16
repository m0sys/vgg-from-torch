import math
from typing import List, Union
import torch.nn as nn
import torch.nn.functional as F
from vgg.base import BaseModel


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


class Vgg(BaseModel):
    def __init__(self, conv_layers, num_classes=100):
        super().__init__()

        self.conv_layers = conv_layers

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)

        ## print(f"shape of x before view: {x.shape}")
        ## print(f"size of x at zero: {x.size(0)}")
        x = x.view(x.size(0), -1)


class Vgg19(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = _create_pooling_layer()

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = _create_pooling_layer()

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = _create_pooling_layer()

        self.conv9 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = _create_pooling_layer()

        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = _create_pooling_layer()

        ## self.fc1 = nn.Linear(in_features=(512 * 7 * 7), out_features=4096)
        ## self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        ## self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Kaiming Init.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool4(x)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool5(x)

        ## print(f"shape of x before view: {x.shape}")
        ## print(f"size of x at zero: {x.size(0)}")
        x = x.view(x.size(0), -1)

        ## x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        ## x = F.dropout(F.relu(self.fc2(x)), training=self.training)

        ## x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        ## x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        # return F.log_softmax(self.fc3(x), dim=1)


_vgg_cfg = {
    # -------------------------------------> 11 weight layers
    'A': [64, 'M', 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    # -------------------------------------> 13 weight layers
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    # -------------------------------------> 16 weight layers
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, '1D-256', 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    # -------------------------------------> 16 weight layers
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    # -------------------------------------> 19 weight layers
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def _build_layers(cfg: List):
    layers = []
    in_channels = 3
    for l in cfg:
        if _is_pooling_layer(l):
            layers.append(_create_pooling_layer())
        else:
            layers.append(_create_conv_layer(l, l))
        in_channels = l
    return nn.Sequential(*layers)


def _is_pooling_layer(l: Union[int, str]) -> bool:
    return l == 'M'


def _create_pooling_layer():
    """Reduce layer area by 50%."""
    return nn.MaxPool2d(stride=2, kernel_size=2)


def _create_conv_layer(in_channels: int, out_channels: int):
    """Return a valid convultion with kernel size 3x3."""
    return F.relu(nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
