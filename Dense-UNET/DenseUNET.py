import torch

import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels, )

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class Tiramisu(nn.Module):
    def downsample(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def __init__(self, in_channels, out_channels):
        super(Tiramisu, self).__init__()
        self.downsample_1 = self.downsample(in_channels=in_channels, out_channels=32)
        # 32 channels
        self.downDense_1 = DenseBlock(32)
        # 40 + 32 channels
        self.maxpool_1 = nn.MaxPool2d(2)

        self.downDense_2 = DenseBlock(72)
        # 40 + 72 channels
        self.maxpool_2 = nn.MaxPool2d(2)

        self.middleDense = DenseBlock(112)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # 40 + 112 channels
        self.upDense_2 = DenseBlock(152)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # 40 + 72 channels
        self.upDense_1 = DenseBlock(112)

        # 40 channels
        self.out = self.final_block(40, 32, out_channels)

    def forward(self, x):
        pre_dense = self.downsample_1(x)
        down_1 = self.downDense_1(pre_dense)
        down_1 = torch.cat([down_1, pre_dense], 1)
        pool_1 = self.maxpool_1(down_1)

        down_2 = self.downDense_2(pool_1)
        down_2 = torch.cat([down_2, pool_1], 1)
        pool_2 = self.maxpool_2(down_2)

        middle = self.middleDense(pool_2)

        up_2 = self.upsample_2(middle)
        up_2 = torch.cat([up_2, down_2], 1)
        up_2 = self.upDense_2(up_2)

        up_1 = self.upsample_1(up_2)
        up_1 = torch.cat([up_1, down_1], 1)
        up_1 = self.upDense_1(up_1)

        output = torch.sigmoid(self.out(up_1))

        return output
