import torch
import torch.nn as nn
import torch.nn.functional as F

from .backwarp import backward_warp

class FeaturePyramidExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[32, 48, 64]):
        super(FeaturePyramidExtractor, self).__init__()

        self.k = len(hidden_layers)
        self.initial = nn.Sequential(nn.Conv2d(input_dim, hidden_layers[0], kernel_size=7, stride=1, padding=3),
                                     nn.ReLU(inplace=True))


        self.A_0 = nn.Sequential(nn.Conv2d(hidden_layers[0], hidden_layers[0], kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(hidden_layers[0], hidden_layers[0], kernel_size=3, padding=1),
                                nn.ReLU())
        self.extra = nn.Sequential(nn.Conv2d(hidden_layers[0], hidden_layers[0], kernel_size=2, stride=2, padding=1), nn.ReLU())
        self.down_A_0 = nn.Sequential(nn.Conv2d(hidden_layers[0], hidden_layers[0], kernel_size=3, stride=2, padding=1),
                                nn.ReLU())

        self.A_1 = nn.Sequential(nn.Conv2d(hidden_layers[0], hidden_layers[1], kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(hidden_layers[1], hidden_layers[1], kernel_size=3, padding=1),
                                nn.ReLU())
        self.down_A_1 = nn.Sequential(nn.Conv2d(hidden_layers[1], hidden_layers[1], kernel_size=3, stride=2, padding=1),
                                     nn.ReLU())

        self.A_2 = nn.Sequential(nn.Conv2d(hidden_layers[1], hidden_layers[2], kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(hidden_layers[2], hidden_layers[2], kernel_size=3, padding=1),
                                nn.ReLU())

    def forward(self, x, num_scales=6):
        x = self.initial(x)

        scale1_1 = self.A_0(x)
        x = F.avg_pool2d(x, kernel_size=2)
        scale1_2 = self.A_1(x)

        scale1 = torch.cat((scale1_1, scale1_2), dim=1)
        scalei_1 = self.A_0(x)
        scalei_2 = self.A_1(self.down_A_0(scale1_1))
        scalei_3 = self.A_2(self.down_A_1(scale1_2))
        scale2 = torch.cat([scalei_1, scalei_2, scalei_3], dim=1)

        scales = [scale2]
        for _ in range(2, num_scales):
            x = F.avg_pool2d(x, kernel_size=2)
            next_scalei_1 = self.A_0(x)
            next_scalei_2 = self.A_1(self.down_A_0(scalei_1))
            scalei_3 = self.A_2(self.down_A_1(scalei_2))
            scalei_1 = next_scalei_1
            scalei_2 = next_scalei_2
            scale_i = torch.cat((scalei_1, scalei_2, scalei_3), dim=1)

            scales.append(scale_i)
        return scales


class ContextExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[16, 18, 20]):

        super(ContextExtractor, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_layers[0], kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers[0], 
                                        out_channels = hidden_layers[0], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels = hidden_layers[0], 
                                        out_channels = hidden_layers[0], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers[0], 
                                        out_channels = hidden_layers[1], 
                                        kernel_size = (3,3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels = hidden_layers[1], 
                                        out_channels = hidden_layers[1], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.ReLU())
        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers[1], 
                                        out_channels = hidden_layers[2], 
                                        kernel_size = (3,3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels = hidden_layers[2], 
                                        out_channels = hidden_layers[2], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.ReLU())
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        scale_1 = self.conv_block1(x)
        scale_2 = self.conv_block2(scale_1)
        scale_3 = self.conv_block3(scale_2)
        return [scale_1, scale_2, scale_3]


class RefinementNet(nn.Module):
    def __init__(self, input_dim=12, hidden_layers=[32, 64, 96], output_dim=4, ce_layers=[0,0,0,0]):
        super(RefinementNet, self).__init__()

        input_dim = 3*2 + ce_layers[0] * 2

        self.down0 = nn.Sequential(nn.Conv2d(input_dim, hidden_layers[0], 3, 2, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_layers[0], hidden_layers[0], 3, 1, 1), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(hidden_layers[0]+(ce_layers[1]*2), hidden_layers[1], 3, 2, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_layers[1], hidden_layers[1], 3, 1, 1), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(hidden_layers[1]+(ce_layers[2]*2), hidden_layers[2], 3, 2, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_layers[2], hidden_layers[2], 3, 1, 1), nn.ReLU(inplace=True))


        self.up0 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden_layers[2], hidden_layers[1], 3, 1, 1), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden_layers[1]*2, hidden_layers[0], 3, 1, 1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden_layers[0]*2, input_dim, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv = nn.Conv2d(input_dim, output_dim, 3, 1, 1)

    def forward(self, img0, img1, ft_0, ft_1, ft_0_s2, ft_1_s2, ft_0_s4, ft_1_s4, c0, c1): 
        s0 = self.down0(torch.cat((img0, img1, backward_warp(c0[0], ft_0), backward_warp(c1[0], ft_1)), dim=1))
        s1 = self.down1(torch.cat((s0, backward_warp(c0[1], ft_0_s2), backward_warp(c1[1], ft_1_s2)), dim=1))
        s2 = self.down2(torch.cat((s1, backward_warp(c0[2], ft_0_s4), backward_warp(c1[2], ft_1_s4)), dim=1))

        x = self.up0(F.interpolate(s2, scale_factor=2.0, mode='bilinear', align_corners=False))
        x = self.up1(F.interpolate(torch.cat((x, s1), dim=1), scale_factor=2.0, mode='bilinear', align_corners=False))
        x = self.up2(F.interpolate(torch.cat((x, s0), dim=1), scale_factor=2.0, mode='bilinear', align_corners=False))
        x = self.conv(x)
        return x