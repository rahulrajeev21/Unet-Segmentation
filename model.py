import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.down1 = downStep(inC=1, outC=64)
        self.down2 = downStep(inC=64, outC=128)
        self.down3 = downStep(inC=128, outC=256)
        self.down4 = downStep(inC=256, outC=512)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottom1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.bottom2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up1 = upStep(inC=1024, outC=512)
        self.up2 = upStep(inC=512, outC=256)
        self.up3 = upStep(inC=256, outC=128)
        self.up4 = upStep(inC=128, outC=64, withReLU=False)

        self.outputlayer = nn.Conv2d(64, 2, 1, stride=1)

    def forward(self, x):
        first = self.down1(x)  # 64
        first_max = self.max_pooling(first)
        second = self.down2(first_max)  # 128
        second_max = self.max_pooling(second)
        third = self.down3(second_max)  # 256
        third_max = self.max_pooling(third)
        fourth = self.down4(third_max)  # 512
        fourth_max = self.max_pooling(fourth)

        bottom_part = self.bottom1(fourth_max)  # 1024
        bottom_part = self.bottom2(bottom_part)  # 1024

        fourth_up = self.up1(bottom_part, fourth)  # 512
        third_up = self.up2(fourth_up, third)  # 256
        second_up = self.up3(third_up, second)  # 128
        first_up = self.up4(second_up, first)  # 64

        x = self.outputlayer(first_up)  # output
        return x


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.downconvs = nn.Sequential(nn.Conv2d(inC, outC, 3, stride=1, padding=1),
                                       nn.BatchNorm2d(outC),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outC, outC, 3, stride=1, padding=1),
                                       nn.BatchNorm2d(outC),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.downconvs(x)
        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        self.deconv = nn.ConvTranspose2d(inC, outC, kernel_size=2, stride=2)
        if (withReLU):
            self.up_convs = nn.Sequential(
                nn.Conv2d(inC, outC, 3, stride=1, padding=1),
                nn.BatchNorm2d(outC),
                nn.ReLU(inplace=True),
                nn.Conv2d(outC, outC, 3, stride=1, padding=1),
                nn.BatchNorm2d(outC),
                nn.ReLU(inplace=True))
        else:
            self.up_convs = nn.Sequential(
                nn.Conv2d(inC, outC, 3, stride=1, padding=1),
                nn.BatchNorm2d(outC),
                nn.Conv2d(outC, outC, 3, stride=1, padding=1),
                nn.BatchNorm2d(outC))

    def forward(self, x, x_down):
        x = self.deconv(x)
        dim_diff_h = x_down.shape[2] - x.shape[2]
        dim_diff_w = x_down.shape[3] - x.shape[3]
        x = F.pad(x, (dim_diff_h, 0, dim_diff_w, 0))
        x_concatenated = torch.cat([x_down, x], dim=1)
        x = self.up_convs(x_concatenated)
        return x
