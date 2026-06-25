import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out


class ResNet15(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.layer1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.layer2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Linear(
            in_features=32,
            out_features=num_classes
        )

    def forward(self, x):

        # [B, 3, 32, 32]
        x = self.stem(x)

        # [B, 32, 32, 32]
        x = self.layer1(x)

        # [B, 32, 16, 16]
        x = self.pool1(x)

        # [B, 32, 16, 16]
        x = self.layer2(x)

        # [B, 32, 8, 8]
        x = self.pool2(x)

        # [B, 32, 1, 1]
        x = self.avgpool(x)

        # [B, 32]
        x = torch.flatten(x, 1)

        # [B, 10]
        x = self.fc(x)

        return x
