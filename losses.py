import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import get_device
from metrics import AverageMeter


class Losses(object):
    def __init__(self, weights=[1,1,1,1], types=['L1', 'L_F', 'Lap', 'Adv']):
        self.weights = weights
        self.types = types
        self.reset()

    def reset(self):
        self.loss_types = {}
        for l in self.types:
            self.loss_types[l] = AverageMeter()
        self.loss_types['total'] = AverageMeter()

    def update(self, values):
        total = 0
        for i in range(len(self.types)):
            self.loss_types[self.types[i]].update(values[i])
            total += values[i] * self.weights[i]
        self.loss_types['total'].update(total)
    
    def get_total(self):
        return self.loss_types['total']


class VGG(nn.Module):
    def __init__(self, model='vgg19', layer=4, reluLayer=True, device='cuda'):
        super(VGG, self).__init__()
        
        # Standard model is VGG-19
        if model == 'vgg16':
            vgg_features = torchvision.models.vgg16(pretrained=True).features.to(device).eval()
        else:
            vgg_features = torchvision.models.vgg19(pretrained=True).features.to(device).eval()

        vgg_features.requires_grad_ = False

        blocks = [i-1 for i, o in enumerate(list(vgg_features)) if isinstance(o, nn.MaxPool2d)]
        feature_index = blocks[layer-1] + int(reluLayer)

        self.vgg = vgg_features[:feature_index]

        # Set in evaluation mode
        for param in self.vgg.parameters():
            param.requires_grad_ = False

        # https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        # Normalize input to ImageNet statistics
  
        self.mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)


    def forward(self, x, gt):
        x = (x - self.mean) / self.std
        gt = (gt - self.mean) / self.std
        vgg_output = self.vgg(x)
        #with torch.no_grad():
        vgg_gt = self.vgg(gt)

        loss = F.mse_loss(vgg_output, vgg_gt.detach())
        return loss


class VGGLossSepConv(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(VGGLossSepConv, self).__init__()

        model = torchvision.models.vgg19(pretrained=True).to(device).eval()

        self.features = nn.Sequential(
            # stop at relu4_4 (-10)
            *list(model.features.children())[:-10]
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(torch.device(device))
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(torch.device(device))
        for param in self.features.parameters():
            param.requires_grad = False
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, output, target):
        output = (output - self.mean) / self.std
        target = (target - self.mean) / self.std
        output_features = self.features(output)
        target_features = self.features(target)

        return self.mse(output_features, target_features)


class L1_Loss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(L1_Loss, self).__init__()

        self.eps = epsilon

    def forward(self, x, gt):
        l1 = torch.add(x, -gt)
        if self.eps != 0:
            l1 = torch.sqrt(l1 * l1 + self.eps * self.eps)
        # Or torch.mean() here
        loss = torch.sum(torch.abs(l1))
        return loss 



# Modified version of https://github.com/tneumann/minimal_glo/blob/master/glo.py

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, device='cuda'):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.from_numpy(kernel).to(torch.float).to(device)
    return kernel

def gaussian_conv(img, kernel):
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)

class LaplacianPyramid(nn.Module):
    def __init__(self, max_level=5):
        super(LaplacianPyramid, self).__init__()
        self.max_level = max_level
        self.kernel = build_gauss_kernel(n_channels=3)

    def forward(self, X):
        pyramids = []
        current = X
        for level in range(self.max_level):
            blurred = gaussian_conv(current, self.kernel)
            # reduce
            next_level = F.avg_pool2d(blurred, 2)
            # expand
            expand_next_level = F.interpolate(next_level, scale_factor=2)
            diff = current - expand_next_level
            pyramids.append(diff)
            current = next_level
        return pyramids

class LaplacianLoss(nn.Module):
    def __init__(self, weights=[1,2,4,8,16], max_level=5, reduction='sum'):
        super(LaplacianLoss, self).__init__()

        self.max_level = max_level
        self.weights = weights
        self.criterion = nn.L1Loss(reduction=reduction)
        self.lap = LaplacianPyramid(max_level=max_level)

    def forward(self, x, y):
        x_lap, y_lap = self.lap(x), self.lap(y)
        scale_factor = 2 ** (self.max_level-1)
        return scale_factor * sum(self.criterion(a, b) * self.weights[i] for i, (a, b) in enumerate(zip(x_lap, y_lap)))

class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt):
        loss_map = (flow - gt) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return loss




class AdversarialLoss(nn.Module):
    def __init__(self, loss='bce', device='cuda'):
        super(AdversarialLoss, self).__init__()

        self.loss = loss
        #self.register_buffer('real_label', torch.tensor(1.0))
        #self.register_buffer('fake_label', torch.tensor(0.0))

        self.real_label = torch.tensor(1.0, device=device)
        self.fake_label = torch.tensor(0.0, device=device)

        if loss == 'bce':
            self.criterion = nn.BCELoss()

        elif loss == 'mse':
            self.criterion = nn.MSELoss()

        elif loss == 'l1':
            self.criterion = nn.L1Loss()

        elif loss == 'hinge':
            self.criterion = nn.ReLU()

    def forward(self, outputs, true_labels, discriminator=True):
        if self.loss == 'hinge':
            if discriminator:
                if true_labels:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        labels = (self.real_label if true_labels else self.fake_label).expand_as(outputs)
        loss = self.criterion(outputs, labels)
        return loss
