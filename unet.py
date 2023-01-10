from typing import List, Tuple

import torch
import torch.functional as F
import torchvision
from torch import nn


class BasicBlock(nn.Module):
    """Basic CNN block"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class EncoderBlock(nn.Module):
    """CNN block + downsample at the end"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = BasicBlock(in_channels, out_channels)
        self.downsample = nn.MaxPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.downsample(x)

        return x


class Encoder(nn.Module):
    "Contracting path"

    def __init__(self, in_channels: int, block_out_channels: Tuple[int]) -> None:
        super().__init__()
        self.downblocks = nn.ModuleList([])

        for out_channels in block_out_channels:
            self.downblocks.append(EncoderBlock(in_channels, out_channels))

            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_maps = []

        for block in self.downblocks:
            x = block(x)
            feat_maps.append(x)

        return feat_maps


# MidBlock = BasicBlock


class DecoderBlock(nn.Module):
    "Upsampling + CNN block"

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        """
        From paper:
            1. upsample: increase H, W and halves the number of channels: C_input_layer --> C_input_layer//2
            2. concatenate with feat map of previous encoder block: C_input_prev_layer = C_input_curr_layer // 2
            3. number of layer input to BasicBlock: C_input_layer//2 + C_input_prev_layer = C_input_layer
        """
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.block = BasicBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        x_prev_cropped = self.crop(feat_map, x)
        x = torch.cat([x_prev_cropped, x], dim=1)

        x = self.block(x)

        return x

    def crop(self, feat_map: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        feat_map = torchvision.transforms.CenterCrop(x.shape[2:])(feat_map)
        return feat_map


class Decoder(nn.Module):
    "Expanding path"

    def __init__(self, in_channels: int, block_out_channels: List[int]) -> None:
        super().__init__()
        self.upblocks = nn.ModuleList([])

        for out_channels in block_out_channels:
            self.upblocks.append(DecoderBlock(in_channels, out_channels))

            in_channels = out_channels

    def forward(self, x: torch.Tensor, feat_maps: List[torch.Tensor]) -> torch.Tensor:

        for block in self.upblocks:
            feat_map = feat_maps.pop()
            x = block(x, feat_map)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block_out_channels: Tuple[int] = (64, 128, 256, 512),
        retain_dims: bool = False,
    ) -> None:
        super().__init__()

        self.retain_dims = retain_dims
        self.midblock_out_channels = block_out_channels[-1] * 2

        self.encoder = Encoder(in_channels, block_out_channels)
        self.midblock = BasicBlock(block_out_channels[-1], self.midblock_out_channels)

        reversed_block_out_channels = list(reversed(block_out_channels))
        self.decoder = Decoder(self.midblock_out_channels, reversed_block_out_channels)

        self.fc = nn.Conv2d(reversed_block_out_channels[-1], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_maps = self.encoder(x)
        out = self.midblock(feat_maps[-1])
        out = self.decoder(out, feat_maps)
        out = self.fc(out)

        if self.retain_dims:
            out = F.interpolate(out, x.shape[2:])

        return out
