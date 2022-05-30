import torch
import torch.nn as nn
import torch.nn.functional as F

from .rdb import RDB_FlowDecoder as RDBCell
from .extractors import FeaturePyramidExtractor as FPE
from .extractors import ContextExtractor as CE
from .forward_warp import ForwardWarp
from .softsplat import FunctionSoftsplat
from .correlation import FunctionCorrelation


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


class Model(nn.Module):
    def __init__(self, frame1, levels=6, device='cuda'):
        super(Model, self).__init__()
        self.device = device
        self.shape = frame1.shape

        self.levels = levels
        fpe_layers = [32, 48, 64]
        self.fpe = FPE(hidden_layers=fpe_layers)
        self.rdb = RDBCell(in_channels=81+2, growthRate=32, num_layer=3, num_blocks=8)
        ce_layers = [32, 48, 64]
        self.context = CE(hidden_layers=ce_layers)
        self.fwarp = ForwardWarp(device=device)
        self.refine = RefinementNet(ce_layers=ce_layers)    


    def forward(self, frame1, frame2, t=0.5):
        n, c, h, w = frame1.shape
        frame1_, frame2_, _, _ = self.normalize(frame1, frame2)

        fmap1, fmap2 = self.fpe(frame1_), self.fpe(frame2_)


        f0_1 = torch.zeros((n, 2, round(h/(2**self.levels)), round(w/(2**self.levels)))).to(self.device)
        f1_0 = torch.zeros((n, 2, round(h/(2**self.levels)), round(w/(2**self.levels)))).to(self.device)

        for i in range(self.levels-2, -1, -1):
            if i != self.levels-2:
                f0_1 = F.interpolate(f0_1, size=(round(h/(2**(i+2))), round(w/(2**(i+2)))), mode='bilinear', align_corners=False) * 2
                f1_0 = F.interpolate(f1_0, size=(round(h/(2**(i+2))), round(w/(2**(i+2)))), mode='bilinear', align_corners=False) * 2
            
            warped_fmap1 = backward_warp(fmap1[i], f1_0)
            warped_fmap2 = backward_warp(fmap2[i], f0_1)            

            cp1 = F.leaky_relu(FunctionCorrelation(fmap1[i], warped_fmap2), negative_slope=0.1, inplace=False)
            cp2 = F.leaky_relu(FunctionCorrelation(fmap2[i], warped_fmap1), negative_slope=0.1, inplace=False)

            inp = torch.cat((torch.cat((cp1, f0_1), dim=1), torch.cat((cp2, f1_0), dim=1)), dim=0)


            res_f = self.rdb(inp)

            res_f0_1, res_f1_0 = torch.split(res_f, [n,n], dim=0)
            
            f0_1 = f0_1 + res_f0_1
            f1_0 = f1_0 + res_f1_0


       
        ft_0_scale2 = F.interpolate(f0_1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2
        ft_1_scale2 = F.interpolate(f1_0, scale_factor=2.0, mode="bilinear", align_corners=False) * 2
        ft_0 = F.interpolate(ft_0_scale2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2
        ft_1 = F.interpolate(ft_1_scale2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2

        ft_0_scale4, ft_1_scale4 = self.reverse_flow(f0_1, f1_0, t)
        ft_0_scale2, ft_1_scale2 = self.reverse_flow(ft_0_scale2, ft_1_scale2, t)
        ft_0, ft_1 = self.reverse_flow(ft_0, ft_1, t)

        
        cmap1, cmap2 = self.context(frame1_), self.context(frame2_)
      
        refined = self.refine(frame1, frame2, ft_0, ft_1, ft_0_scale2, ft_1_scale2, ft_0_scale4, ft_1_scale4, cmap1, cmap2)
        res = torch.tanh(refined[:, :3])
        mask = torch.sigmoid(refined[:, 3:4])

        w1, w2 = (1-t) * mask, t * (1 - mask)
        merged_img = (w1 * backward_warp(frame1, ft_0) + w2 * backward_warp(frame2, ft_1)) / (w1 + w2 + 1e-8)
        image_t = merged_img + res
        image_t = torch.clamp(image_t, 0, 1)
        
        return image_t, [ft_0, ft_1]

        


    @staticmethod
    def normalize(frame1, frame2):
        _, _, H, W = frame1.shape
        stacked_frames = torch.cat((frame1, frame2), dim=2)
        mean = torch.mean(stacked_frames, dim=(2,3), keepdim=True)
        std = torch.std(stacked_frames, dim=(2,3), keepdim=True)
        normalized = (stacked_frames - mean) / (std + 1e-8)
        frame1, frame2 = torch.split(normalized, H, dim=2)
        return frame1, frame2, mean, std

    @staticmethod
    def unnormalize(frame, mean, std):
        frame = frame * std + mean
        return frame

    def reverse_flow(self, f0_1, f1_0, t):
        one_size = f0_1.size()
        ones0 = torch.ones(one_size, requires_grad=True, device=self.device)
        ones1 = torch.ones(one_size, requires_grad=True, device=self.device)

        flow_forward = FunctionSoftsplat(tenInput=f0_1, tenFlow=t*f0_1, tenMetric=None, strType='summation')
        flow_backward = FunctionSoftsplat(tenInput=f1_0, tenFlow=(1-t)*f1_0, tenMetric=None, strType='summation')
        
        norm0 = FunctionSoftsplat(tenInput=ones0, tenFlow=t*f0_1, tenMetric=None, strType='summation')
        norm1 = FunctionSoftsplat(tenInput=ones1, tenFlow=(1-t)*f1_0, tenMetric=None, strType='summation')

        flow_t0 = -(1-t) * t*flow_forward + t * t * flow_backward
        flow_t1 = (1-t) * (1-t)*flow_forward - t * (1-t) * flow_backward

        norm = (1-t)*norm0 + t*norm1
        mask_ = (norm.detach() > 0).type(norm.type())

        flow_t0 = (1-mask_) * flow_t0 + mask_ * (flow_t0.clone() / (norm.clone() + (1-mask_)))
        flow_t1 = (1-mask_) * flow_t1 + mask_ * (flow_t1.clone() / (norm.clone() + (1-mask_)))

        return flow_t0, flow_t1


backward_grid = {}

def backward_warp(frame, flow, device='cuda'):
    if str(flow.shape) not in backward_grid:
        tenHor = torch.linspace(-1.0 + (1.0 / flow.shape[3]), 1.0 - (1.0 / flow.shape[3]), flow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, flow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / flow.shape[2]), 1.0 - (1.0 / flow.shape[2]), flow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, flow.shape[3])

        backward_grid[str(flow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(device)

    flow = torch.cat([ flow[:, 0:1, :, :] / ((frame.shape[3] - 1.0) / 2.0), flow[:, 1:2, :, :] / ((frame.shape[2] - 1.0) / 2.0) ], 1)

    return F.grid_sample(input=frame.float(), grid=(backward_grid[str(flow.shape)] + flow.float()).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)


def compute_cost_volume(feat1, feat2, max_disp=4):
    """
    only implemented for:
        kernel_size = 1
        stride1 = 1
        stride2 = 1
    """

    _, _, height, width = feat1.size()
    num_shifts = 2 * max_disp + 1
    feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, axis=1)
    return cost_volume
