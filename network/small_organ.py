import torch.nn as nn
import torch

from monai.networks.blocks import Convolution, MaxAvgPool, ResidualSELayer, SimpleASPP, UpSample
from monai.networks.layers.factories import Act, Norm
from monai.utils import ensure_tuple_rep


class ConvBNActBlock(nn.Module):
    """Two convolution layers with batch norm, leaky relu, dropout and SE block"""

    def __init__(self, in_channels, out_channels, dropout_p, spatial_dims: int = 3):
        super().__init__()

        self.conv_conv_se = nn.Sequential(
            Convolution(spatial_dims, in_channels, out_channels, kernel_size=3, norm=Norm.BATCH, act=Act.LEAKYRELU),
            nn.Dropout(dropout_p),
            Convolution(spatial_dims, out_channels, out_channels, kernel_size=3, norm=Norm.BATCH, act=Act.LEAKYRELU),
            ResidualSELayer(spatial_dims=spatial_dims, in_channels=out_channels, r=2),
        )

    def forward(self, x):
        return self.conv_conv_se(x)


class DownBlock(nn.Module):
    """
    Downsampling with a concatenation of max-pool and avg-pool, followed by ConvBNActBlock
    """

    def __init__(self, in_channels, out_channels, dropout_p, spatial_dims: int = 3):
        super().__init__()
        self.max_avg_pool = MaxAvgPool(spatial_dims=spatial_dims, kernel_size=2)
        self.conv = ConvBNActBlock(2 * in_channels, out_channels, dropout_p, spatial_dims=spatial_dims)
        # if (in_channels * 2) != out_channels:
        #     self.downsample = Convolution(spatial_dims, in_channels * 2, out_channels, norm=Norm.BATCH, act=Act.LEAKYRELU, bias=False)
        # else:
        #     self.downsample = lambda x: x

    def forward(self, x):
        x_pool = self.max_avg_pool(x)
        # residual = self.downsample(x_pool)
        return self.conv(x_pool)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBNActBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True, dropout_p=0.5, spatial_dims: int = 3):
        super().__init__()
        self.up = UpSample(spatial_dims, in_channels1, in_channels2, scale_factor=2)
        self.conv = ConvBNActBlock(in_channels2 * 2, out_channels, dropout_p, spatial_dims=spatial_dims)
        # if (in_channels2 * 2) != out_channels:
        #     self.downsample = Convolution(spatial_dims, in_channels2 * 2, out_channels, norm=Norm.BATCH, act=Act.LEAKYRELU, bias=False)
        # else:
        #     self.downsample = lambda x: x

    def forward(self, x1, x2):
        x_cat = torch.cat([x2, self.up(x1)], dim=1)
        # residual = self.downsample(x_cat)
        return self.conv(x_cat)


class organNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        feature_channels=(32, 40, 48, 56, 64),
        dropout=(0.0, 0.0, 0.0, 0.0, 0.0),
        bilinear: bool = True,
    ):
        """
        Args:
            spatial_dims: dimension of the operators. Defaults to 2, i.e., using 2D operators
                for all operators, for example, using Conv2D for all the convolutions.
                It should be 2 for 3D images
            in_channels: number of channels of the input image. Defaults to 1.
            out_channels: number of segmentation classes (2 for foreground/background segmentation).
                Defaults to 2.
            feature_channels: number of intermediate feature channels
                (must have 5 elements corresponding to five conv. stages).
                Defaults to (32, 64, 128, 256, 512).
            dropout: a sequence of 5 dropout ratios. Defaults to (0.0, 0.0, 0.3, 0.4, 0.5).
            bilinear: whether to use bilinear upsampling. Defaults to True.
        """
        super().__init__()
        ft_chns = ensure_tuple_rep(feature_channels, 5)

        f0_half = int(ft_chns[0] / 2)
        f1_half = int(ft_chns[1] / 2)
        f2_half = int(ft_chns[2] / 2)
        f3_half = int(ft_chns[3] / 2)

        self.in_conv = ConvBNActBlock(in_channels, ft_chns[0], dropout[0], spatial_dims)
        self.down1 = DownBlock(ft_chns[0], ft_chns[1], dropout[1], spatial_dims)
        self.down2 = DownBlock(ft_chns[1], ft_chns[2], dropout[2], spatial_dims)
        self.down3 = DownBlock(ft_chns[2], ft_chns[3], dropout[3], spatial_dims)
        self.down4 = DownBlock(ft_chns[3], ft_chns[4], dropout[4], spatial_dims)

        self.bridge0 = Convolution(spatial_dims, ft_chns[0], f0_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)
        self.bridge1 = Convolution(spatial_dims, ft_chns[1], f1_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)
        self.bridge2 = Convolution(spatial_dims, ft_chns[2], f2_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)
        self.bridge3 = Convolution(spatial_dims, ft_chns[3], f3_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)

        self.up1 = UpBlock(ft_chns[4], f3_half, ft_chns[2], bilinear, dropout[3], spatial_dims)
        self.up2 = UpBlock(ft_chns[2], f2_half, ft_chns[1], bilinear, dropout[2], spatial_dims)
        self.up3 = UpBlock(ft_chns[1], f1_half, ft_chns[0], bilinear, dropout[1], spatial_dims)
        self.up4 = UpBlock(ft_chns[0], f0_half, ft_chns[0], bilinear, dropout[0], spatial_dims)

        self.out_conv = Convolution(spatial_dims, ft_chns[0], out_channels, conv_only=True)

        self.initialize()

    def forward(self, x):

        x0 = self.in_conv(x)
        x0b = self.bridge0(x0)
        x1 = self.down1(x0)
        x1b = self.bridge1(x1)
        x2 = self.down2(x1)
        x2b = self.bridge2(x2)
        x3 = self.down3(x2)
        x3b = self.bridge3(x3)
        x4 = self.down4(x3)

        x = self.up1(x4, x3b)
        x = self.up2(x, x2b)
        x = self.up3(x, x1b)
        x = self.up4(x, x0b)
        output = self.out_conv(x)

        return output

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)


if __name__ == '__main__':
    import torch
    import os
    from torchsummary import summary
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    img = torch.randn((1, 1, 96, 96, 96))
    model = organNet().to(torch.device('cpu'))
    # out = model(img)
    summary(model, (1, 96, 96, 96), device='cpu')