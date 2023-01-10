from typing import List, Optional, Type, Union

import torch
from torch import nn


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    """Figure 5 -- Left."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        skip: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.cnn1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.cnn2 = conv3x3(out_channels, out_channels * self.expansion)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = skip if skip else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 3x3, 64
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3, 64
        out = self.conv2(out)
        out = self.bn2(out)

        # The second non-linearity is after y = F + x
        out += self.skip(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Figure 5 -- Right
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        skip: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # 1x1, 64
        self.cnn1 = conv1x1(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3, 64
        self.cnn2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1, 256 (= 64 * 4)
        self.cnn3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.skip = skip if skip else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        # The third non-linearity is after y = F + x (see Fig. 5)
        out += self.skip(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block_class: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 1000,
        init_residual_weights: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = 64

        # 7x7 conv, 64, /2 + max pool /2 (see Table 1.)
        self.layer0 = nn.Sequential(
            [
                nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        self.layer1 = self._make_layer(
            block=block_class, curr_out_channels=64, num_blocks=layers[0], stride=2
        )

        self.layer2 = self._make_layer(
            block=block_class, curr_out_channels=128, num_blocks=layers[0], stride=2
        )

        self.layer3 = self._make_layer(
            block=block_class, curr_out_channels=256, num_blocks=layers[0], stride=2
        )

        self.layer4 = self._make_layer(
            block=block_class, curr_out_channels=512, num_blocks=layers[0], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_class.expansion, num_classes)

        self.initialize_weights(init_residual_weights)

    def initialize_weights(self, init_residual_weights: bool) -> None:

        """We initialize the weights as in [13] --  Delving deep into rectifiers:
        Surpassing human-level performance on imagenet classification.

        This corresponds to torch.nn.init.kaiming_normal_
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        """Paper [https://arxiv.org/pdf/1706.02677.pdf]: For BN layers, the learnable 
            scaling coefficient γ is initialized to be 1, except for each residual 
            block's last BN where γ is initialized to be 0. Setting γ = 0 in the last 
            BN of each residual block causes the forward/backward signal initially to
            propagate through the identity shortcut of ResNets, which we found to ease 
            optimization at the start of training.
        """
        if init_residual_weights:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        curr_out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:

        skip = None

        if stride != 1 or self.in_channels != curr_out_channels * block.expansion:
            skip = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    curr_out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(curr_out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=curr_out_channels,
                stride=stride,
                downsample=skip,
            )
        )

        self.in_channels = curr_out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=curr_out_channels,
                    downsample=None,
                )
            )

        return nn.Sequential(*layers)


def ResNet34(num_classes: int, channels: int = 3) -> ResNet:
    # See Table 1
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, channels)


def ResNet50(num_classes: int, channels: int = 3) -> ResNet:
    # See Table 1
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)
