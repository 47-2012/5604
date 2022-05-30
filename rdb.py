import torch
import torch.nn as nn 


class Network(nn.Module):
    def __init__(self, in_channels=16, layers=[32,64,128,96,64,32], output=4, device='cuda'):

        super(Network, self).__init__()

        self.layer1 = BottleneckBlockDil(in_channels, layers[0], padding=1)
        #layer2 = BottleneckBlock(layers[0], layers[0], stride=1)
        #layers = (layer1, layer2)
        #self.layer1 = nn.Sequential(*layers)

        self.layer2 = BottleneckBlockDil(layers[0], layers[1], padding=1)
        #layer2 = BottleneckBlock(layers[1], layers[1], stride=1)
        #layers = (layer1, layer2)
        #self.layer2 = nn.Sequential(*layers)

        self.layer3 = BottleneckBlockDil(layers[1], layers[2], padding=2, dilation=2)
        #layer2 = BottleneckBlock(layers[2], layers[2], stride=1, padding=2, dilation=2)
        #layers = (layer1, layer2)
        #self.layer3 = nn.Sequential(*layers)

        self.layer4 = BottleneckBlockDil(layers[2], layers[3], padding=2, dilation=2)
        #layer2 = BottleneckBlock(layers[3], layers[3], stride=1, padding=2, dilation=2)
        #layers = (layer1, layer2)
        #self.layer4 = nn.Sequential(*layers)

        self.layer5 = BottleneckBlockDil(layers[3], layers[4], padding=2, dilation=2)
        #layer2 = BottleneckBlock(layers[4], layers[4], stride=1, padding=2, dilation=2)
        #layers = (layer1, layer2)
        #self.layer5 = nn.Sequential(*layers)

        self.layer6 = BottleneckBlockDil(layers[4], layers[5], padding=1)
        #layer2 = BottleneckBlock(layers[5], layers[5], stride=1)
        #layers = (layer1, layer2)
        #self.layer6 = nn.Sequential(*layers)

        self.layer7 = BottleneckBlockDil(layers[5], output, padding=1)
        #layer2 = BottleneckBlock(output, output, stride=1)
        #layers = (layer1, layer2)
        #self.layer7 = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

class Network_RES(nn.Module):
    def __init__(self, in_channels=16, layers=[32,64,128,96,64,32], output=4, device='cuda'):

        super(Network_RES, self).__init__()

        self.seq = nn.Sequential(
            ResidualBlock(in_channels, layers[0], down=True),
            ResidualBlock(layers[0], layers[1], down=True),
            ResidualBlock(layers[1], layers[2], down=True), #dil=2, pad=2
            ResidualBlock(layers[2], layers[3], down=True), #dil=2, pad=2
            ResidualBlock(layers[3], layers[4], down=True), #dil=2, pad=2
            ResidualBlock(layers[4], layers[5], down=True),
            ResidualBlock(layers[5], output, down=True)
            #nn.PReLU()
            #nn.Tanh()
        )

    def forward(self, x):
        return self.seq(x)

class FPE_RES(nn.Module):
    def __init__(self, hidden_layers=[32,64,96], input_dim=3,k=3, device='cuda'):
        super(FPE_RES, self).__init__()
        self.k = k
        self.conv1 = ResidualBlock(input_dim, hidden_layers[0], bias=False, down=True)
        self.conv2 = ResidualBlock(hidden_layers[0], hidden_layers[0], bias=False)

        self.conv3 = ResidualBlock(hidden_layers[0], hidden_layers[1], bias=False, stride=2)
        self.conv4 = ResidualBlock(hidden_layers[1], hidden_layers[1], bias=False)

        self.conv5 = ResidualBlock(hidden_layers[1], hidden_layers[2], bias=False, stride=2)
        self.conv6 = ResidualBlock(hidden_layers[2], hidden_layers[2], bias=False)

        if k==4:
            self.conv7 = ResidualBlock(hidden_layers[2], hidden_layers[3], bias=False, stride=2)
            self.conv8 = ResidualBlock(hidden_layers[3], hidden_layers[3], bias=False, stride=1)

    def forward(self, x):
        scale1 = self.conv2(self.conv1(x))
        scale2 = self.conv4(self.conv3(scale1))
        scale3 = self.conv6(self.conv5(scale2))
        if self.k == 4:
            scale4 = self.conv8(self.conv7(scale3))
            return [scale1, scale2, scale3, scale4]
        return [scale1, scale2, scale3]


class FeatureExtractorRes(nn.Module):
    def __init__(self, hidden_layers=[32,64,96], input_dim=3, k=3, device='cuda'):
        super(FeatureExtractorRes, self).__init__()
        self.k = k

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)


        #stride layer 1 is 1
        layer1 = BottleneckBlock(input_dim, hidden_layers[0], down=True)
        layer2 = BottleneckBlock(hidden_layers[0], hidden_layers[0], stride=1)
        layers = (layer1, layer2)
        self.layer1 = nn.Sequential(*layers)

        # stride layer 2 is 2
        layer1 = BottleneckBlock(hidden_layers[0], hidden_layers[1], stride=2)
        layer2 = BottleneckBlock(hidden_layers[1], hidden_layers[1], stride=1)
        layers = (layer1, layer2)
        self.layer2 = nn.Sequential(*layers)

        layer1 = BottleneckBlock(hidden_layers[1], hidden_layers[2], stride=2)
        layer2 = BottleneckBlock(hidden_layers[2], hidden_layers[2], stride=1)
        layers = (layer1, layer2)
        self.layer3 = nn.Sequential(*layers)

        if k==4:
            layer1 = BottleneckBlock(hidden_layers[2], hidden_layers[3], stride=2)
            layer2 = BottleneckBlock(hidden_layers[3], hidden_layers[3], stride=1)
            layers = (layer1, layer2)
            self.layer4 = nn.Sequential(*layers)

    def forward(self, x):
        #x = self.relu1(self.conv1(x))
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        if self.k == 4:
            layer4 = self.layer4(layer3)
            return [layer1, layer2, layer3, layer4]
        return [layer1, layer2, layer3]



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='none', stride=1, dilation=1, bias=True, down=False):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0, bias=bias)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride, bias=bias)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.PReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        self.down = False
        
        if stride > 1 or dilation > 1 or down==True:  
            self.down = True
            self.downsample = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias, stride=stride)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.down:
            x = self.downsample(x)


        return self.relu(x+y)

class BottleneckBlockDil(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='none', stride=1, padding=0, dilation=1):
        super(BottleneckBlockDil, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=padding, dilation=dilation)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.conv_projection = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()



    def forward(self, x):
        y = x
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))

        x = self.conv_projection(x)
        return self.relu(x+y)







class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='none', kernel_size=3, padding=1, dilation=1, stride=1, bias=True, down=False):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.PReLU()

        self.down=False
        if stride > 1 or down==True:
            self.down = True
            self.conv_projection = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias, stride=stride)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()


    def forward(self, x):
        y = x
        #y = self.relu(self.norm1(self.conv1(y)))
        #y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        
        if self.down:
            x = self.conv_projection(x)
    
        return self.relu(x+y)

class FeaturePyramidExtractor(nn.Module):
    def __init__(self, hidden_layers, input_dim=3, k=3, device='cuda'):

        super(FeaturePyramidExtractor, self).__init__()
        self.k = k

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels = input_dim, 
                                        out_channels = hidden_layers[0], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.PReLU(),
                                        nn.Conv2d(in_channels = hidden_layers[0], 
                                        out_channels = hidden_layers[0], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.PReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers[0], 
                                        out_channels = hidden_layers[1], 
                                        kernel_size = (3,3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.PReLU(),
                                        nn.Conv2d(in_channels = hidden_layers[1], 
                                        out_channels = hidden_layers[1], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.PReLU())
        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers[1], 
                                        out_channels = hidden_layers[2], 
                                        kernel_size = (3,3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.PReLU(),
                                        nn.Conv2d(in_channels = hidden_layers[2], 
                                        out_channels = hidden_layers[2], 
                                        kernel_size = (3,3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=False),
                                        nn.PReLU())
        if k == 4:
            self.conv_block4 = nn.Sequential(nn.Conv2d(in_channels = hidden_layers[2],
                                            out_channels = hidden_layers[3],
                                            kernel_size=(3,3),
                                            stride=(2,2), 
                                            padding=(1,1), 
                                            bias=False),
                                            nn.PReLU(),
                                            nn.Conv2d(in_channels = hidden_layers[3], 
                                            out_channels = hidden_layers[3],
                                            kernel_size=(3,3),
                                            stride=(1,1),
                                            padding=(1,1),
                                            bias=False),
                                            nn.PReLU())
        
    def forward(self, x):
        scale_1 = self.conv_block1(x)
        scale_2 = self.conv_block2(scale_1)
        scale_3 = self.conv_block3(scale_2)
        if self.k == 4:
            scale_4 = self.conv_block4(scale_3)
            return [scale_1, scale_2, scale_3, scale_4]
        return [scale_1, scale_2, scale_3]


class Conv2dDown(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=2, padding=1):
        super(Conv2dDown, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.PReLU(),
                                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
                                nn.PReLU())

    def forward(self, x):
        return self.seq(x)

class Conv2dUp(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(Conv2dUp, self).__init__()
        self.seq = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.PReLU(),
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.PReLU(),
                                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        return self.seq(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat((x, self.relu(self.conv(x))), 1)


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(DenseLayer(in_channels_, growthRate))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = nn.Conv2d(in_channels_, in_channels, kernel_size=1)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        #out += x
        return out + x


class RDB_FlowDecoder(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks):
        super(RDB_FlowDecoder, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer))
        self.conv1x1 = nn.Conv2d(num_blocks * in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.flow = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        #self.imp_metric = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)



    def forward(self, x):
        out = []
        #n,c,h_,w = x.shape
        #out = torch.empty((self.num_blocks, n, c, h_, w)).to('cuda')
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
            #out[:,i*c:i*c+c, :, :] = h
            #out[i] = h
        out = torch.cat(out, dim=1)
        #out = out.view((n, self.num_blocks*c, h_, w))
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return self.flow(out)

import os
import subprocess
def get_gpu_usage():
    pid = os.getpid()
    process = subprocess.run("nvidia-smi | grep {} | sed 's/ /\\n/g' | grep MiB | grep -o [0-9]*".format(pid),
        encoding='utf-8', shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    usage = int(process.stdout)
    return usage
#model = RDB_FlowDecoder(in_channels=64, growthRate=32, num_layer=3, num_blocks=15)
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))
