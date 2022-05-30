import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_psnr(img1, img2):
    'image range [0-1]'
    mse = torch.mean((img1*255.0 - img2*255.0).pow(2))
    return 10 * torch.log10(255.0**2 / mse)
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = Variable(_3D_window.expand(1, channel, window_size, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, matlab=True):
    'image range [0-1]'
    L = 1

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        if matlab:
            window = create_window_3d(real_size, channel=1).to(img1.device)
            img1 = img1.unsqueeze(1)
            img2 = img2.unsqueeze(1)
            # Channel is set to 1 since we consider color images as volumetric images
        else:
            window = create_window(real_size, channel=channel).to(img1.device)


    if matlab:
        mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
        mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    else:
        mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
        mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    if matlab:
        sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
        sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
        sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2
    else:
        sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class Metrics(object):
    def __init__(self):
        self.psnrs = AverageMeter()
        self.ssims = AverageMeter()
        self.ssims_matlab = AverageMeter()
        self.interpolation_error = AverageMeter()
        self.ie_rife = AverageMeter()
        self.lpips = None
    
    def reset(self):
        self.psnrs.reset()
        self.ssims.reset()
        self.ssims_matlab.reset()
        self.interpolation_error.reset()
        self.lpips = None

    def eval_metrics(self, pred, gt, lpips=None):
        for i in range(gt.size(0)):
            image1, image2 = pred[i], gt[i]
            self.psnrs.update(get_psnr(image1, image2).cpu())
            #if get_psnr(image1, image2) < 30:
            #    print(i, get_psnr(image1, image2))
            self.ssims.update(ssim(image1.unsqueeze(0), image2.unsqueeze(0), matlab=False).cpu())
            self.ssims_matlab.update(ssim(image1.unsqueeze(0), image2.unsqueeze(0), matlab=True).cpu())
            #self.interpolation_error.update(torch.sqrt(F.mse_loss(image1, image2)).cpu()) 
            self.interpolation_error.update(torch.mean(torch.abs((image1*255.0 - image2*255.0))).cpu())
            self.ie_rife.update(np.abs((np.round((image1*255).cpu().numpy()).astype('uint8') - image2.cpu().numpy()*255.0)).mean())


