import math
from typing import List, Union

import torch.nn as nn
import torch.nn.functional as F

from vgg.base import BaseModel


class _Vgg(BaseModel):
    def __init__(self, conv_layers, num_classes=100, default=True):
        super().__init__()

        self.conv_layers = conv_layers

        fc_units = 512 if default else 4096
        num_flatten_params = 512 * 1 * 1 if default else 512 * 7 * 7

        self.fc_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_flatten_params, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = _flatten(x)
        return self.fc_layers(x)


def _flatten(x):
    return x.view(x.size(0), -1)


class Vgg11(_Vgg):
    def __init__(self, num_classes=100, default=True):
        conv_layers = _build_layers(_vgg_cfg["A"])
        super().__init__(
            conv_layers=conv_layers, num_classes=num_classes, default=default
        )


class Vgg13(_Vgg):
    def __init__(self, num_classes=100, default=True):
        conv_layers = _build_layers(_vgg_cfg["B"])
        super().__init__(conv_layers=conv_layers, num_classes=num_classes)


class Vgg16(_Vgg):
    def __init__(self, num_classes=100, default=True):
        conv_layers = _build_layers(_vgg_cfg["D"])
        super().__init__(conv_layers=conv_layers, num_classes=num_classes)


class Vgg19(_Vgg):
    def __init__(self, num_classes=100, default=True):
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
