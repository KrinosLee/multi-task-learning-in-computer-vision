from collections import OrderedDict
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=2, sampling ='down', mode='normal'):
        super(ResidualBlock,self).__init__()
        self.sampling = sampling
        self.bottle_scale = 4
        self.bottle_channels = int(in_channels / self.bottle_scale)
        self.size = size

        self.blocks = nn.ModuleList()

        if mode == 'bottle':
            for i in range(self.size):
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.bottle_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.bottle_channels),
                    nn.ReLU(True),
                    nn.Conv2d(self.bottle_channels, self.bottle_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.bottle_channels),
                    nn.ReLU(True),
                    nn.Conv2d(self.bottle_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(in_channels)
                ))
        else:
            for i in range(self.size):
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels)
                ))

        if sampling == 'down':
            self.sampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2,stride=2),
            )
        else:
            self.sampler =nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def forward(self,x):
        residual = x
        for layer in self.blocks:
            x = layer(x)
            x += residual
        x = self.sampler(x)
        return x
