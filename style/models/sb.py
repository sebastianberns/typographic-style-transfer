#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _Basemodel


class _BasicConv2d(nn.Module):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py#L317
    # change: conv with bias
    def __init__(self, in_channels, out_channels, **kwargs):
        super(_BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# SB CNN 1
class SB_CNN_1(_Basemodel):
    r"""
    Separate columns for both inputs a,b
    Two convolutions (3x3) that maintain the image size (stride 1, padding 2)
        Padding adds zeros all around. Whenever zero represents white paper (no ink)
        this is a neutral operation since it just adds some whitespace.
        -> make sure in data set that 0.0 stands for white and 1.0 for black
        -> input data has to be inverted
    
    Columns are concatenated and passed through two fully connected aggregation layers
    """
    def __init__(self):
        super(SB_CNN_1, self).__init__()
        
        self.conv1 = nn.Conv2d(1,  8, kernel_size=3, stride=1, padding=1) # 3x3x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 3x3x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*16*16*2, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.out = nn.Linear(512, 1)
    
    def forward(self, x):
        a, b = x
        
        # 64x64x1 -> 64x64x8
        a = F.relu(self.conv1(a))
        b = F.relu(self.conv1(b))
        
        # 64x64x8 -> 32x32x8
        a = self.pool(a)
        b = self.pool(b)
        
        # 32x32x8 -> 32x32x16
        a = F.relu(self.conv2(a))
        b = F.relu(self.conv2(b))
        
        # 32x32x16 -> 16x16x16
        a = self.pool(a)
        b = self.pool(b)
        
        # 16x16x16 -> 1x4096
        a = a.view(-1, 16*16*16)
        b = b.view(-1, 16*16*16)
        
        # concatenate a,b horizontally
        x = torch.cat((a,b),1)     # 4096 + 4096 -> 8192
        
        x = F.relu(self.fc1(x))    # 8192 -> 2048
        x = F.relu(self.fc2(x))    # 2048 -> 512
        
        x = F.sigmoid(self.out(x)) # 512 -> 1
        return x


# SB CNN 2
# Model 1 with batchnorm
class SB_CNN_2(_Basemodel):
    r"""
    Separate columns for both inputs a,b
    Two convolutions (3x3) that maintain the image size (stride 1, padding 1)
    Columns are concatenated and passed through two fully connected aggregation layers
    """
    def __init__(self):
        super(SB_CNN_2, self).__init__()
        
        self.conv1 = _BasicConv2d(1,  8, kernel_size=3, stride=1, padding=1) # 3x3x8
        self.conv2 = _BasicConv2d(8, 16, kernel_size=3, stride=1, padding=1) # 3x3x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*16*16*2, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.out = nn.Linear(512, 1)
    
    def forward(self, x):
        a, b = x
        
        # 64x64x1 -> 64x64x8
        a = self.conv1(a)
        b = self.conv1(b)
        
        # 64x64x8 -> 32x32x8
        a = self.pool(a)
        b = self.pool(b)
        
        # 32x32x8 -> 32x32x16
        a = self.conv2(a)
        b = self.conv2(b)
        
        # 32x32x16 -> 16x16x16
        a = self.pool(a)
        b = self.pool(b)
        
        # 16x16x16 -> 1x4096
        a = a.view(-1, 16*16*16)
        b = b.view(-1, 16*16*16)
        
        # concatenate a,b horizontally
        x = torch.cat((a,b),1)     # 4096 + 4096 -> 8192
        
        x = F.relu(self.fc1(x))    # 8192 -> 2048
        x = F.relu(self.fc2(x))    # 2048 -> 512
        
        x = F.sigmoid(self.out(x)) # 512 -> 1
        return x


# SB CNN 3
# Model 2 with 5x5 filters
class SB_CNN_3(_Basemodel):
    r"""
    Separate columns for both inputs a,b
    Two convolutions (5x5) that maintain the image size (stride 1, padding 2)
    Columns are concatenated and passed through two fully connected aggregation layers
    """
    def __init__(self):
        super(SB_CNN_3, self).__init__()
        
        self.conv1 = _BasicConv2d(1,  8, kernel_size=5, stride=1, padding=2) # 5x5x8
        self.conv2 = _BasicConv2d(8, 16, kernel_size=5, stride=1, padding=2) # 5x5x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*16*16*2, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.out = nn.Linear(512, 1)
    
    def forward(self, x):
        a, b = x
        
        # 64x64x1 -> 64x64x8
        a = self.conv1(a)
        b = self.conv1(b)
        
        # 64x64x8 -> 32x32x8
        a = self.pool(a)
        b = self.pool(b)
        
        # 32x32x8 -> 32x32x16
        a = self.conv2(a)
        b = self.conv2(b)
        
        # 32x32x16 -> 16x16x16
        a = self.pool(a)
        b = self.pool(b)
        
        # 16x16x16 -> 1x4096
        a = a.view(-1, 16*16*16)
        b = b.view(-1, 16*16*16)
        
        # concatenate a,b horizontally
        x = torch.cat((a,b),1)     # 4096 + 4096 -> 8192
        
        x = F.relu(self.fc1(x))    # 8192 -> 2048
        x = F.relu(self.fc2(x))    # 2048 -> 512
        
        x = F.sigmoid(self.out(x)) # 512 -> 1
        return x


# SB CNN 4
# Model 3 with fc dropout and more filters in conv2
class SB_CNN_4(_Basemodel):
    r"""
    Separate columns for both inputs a,b
    Two convolutions (5x5) that maintain the image size (stride 1, padding 2)
    Columns are concatenated and passed through two fully connected aggregation layers
    """
    def __init__(self):
        super(SB_CNN_4, self).__init__()
        
        self.features = nn.Sequential(
            # conv1 5x5x8
            nn.Conv2d(1, 8, bias=True, kernel_size=5, stride=1, padding=2), # 64x64x1 -> 64x64x8
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64x8 -> 32x32x8
            
            # conv2 5x5x4
            nn.Conv2d(8, 32, bias=True, kernel_size=5, stride=1, padding=2), # 32x32x8 -> 32x32x32
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32x32 -> 16x16x32
        )
        self.classifier = nn.Sequential(
            # fc1
            nn.Dropout(),
            nn.Linear(16*16*32*2, 8192), # 16384 -> 8192
            nn.ReLU(inplace=True),
            
            #fc2
            nn.Dropout(),
            nn.Linear(8192, 4096), # 8192 -> 4096
            nn.ReLU(inplace=True),
            
            # output
            nn.Linear(4096, 1), # 4096 -> 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        a, b = x
        N = a.size(0)
        
        # 64x64x1 -> 16x16x32
        a = self.features(a)
        b = self.features(b)
        
        # 16x16x32 -> 8192
        a = a.view(N, 16*16*32)
        b = b.view(N, 16*16*32)
        
        # 8192 + 8192 -> 16384
        x = torch.cat((a,b),1)
        
        x = self.classifier(x)
        return x


# SB CNN 5
# Model 3 with conv dropout
class SB_CNN_5(_Basemodel):
    r"""
    Separate columns for both inputs a,b
    Two convolutions (5x5) that maintain the image size (stride 1, padding 2)
    Columns are concatenated and passed through two fully connected aggregation layers
    """
    def __init__(self):
        super(SB_CNN_5, self).__init__()
        
        self.features = nn.Sequential(
            # conv1 5x5x8
            nn.Conv2d(1, 8, bias=True, kernel_size=5, stride=1, padding=2), # 64x64x1 -> 64x64x8
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64x8 -> 32x32x8
            nn.Dropout2d(p=0.2),
            
            # conv2 5x5x2
            nn.Conv2d(8, 16, bias=True, kernel_size=5, stride=1, padding=2), # 32x32x8 -> 32x32x16
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32x16 -> 16x16x16
            nn.Dropout2d(p=0.2)
        )
        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(16*16*16*2, 2048), # 8192 -> 2048
            nn.ReLU(inplace=True),
            
            #fc2
            nn.Linear(2048, 512), # 2048 -> 512
            nn.ReLU(inplace=True),
            
            # output
            nn.Linear(512, 1), # 512 -> 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        a, b = x
        N = a.size(0)
        
        # 64x64x1 -> 16x16x16
        a = self.features(a)
        b = self.features(b)
        
        # 16x16x16 -> 4096
        a = a.view(N, 16*16*16)
        b = b.view(N, 16*16*16)
        
        # 4096 + 4096 -> 8192
        x = torch.cat((a,b),1)
        
        # 8192 -> 1
        x = self.classifier(x)
        return x


# SB CNN 6
# Model 3 in sequential form
class SB_CNN_6(_Basemodel):
    r"""
    Separate columns for both inputs a,b
    Two convolutions (5x5) that maintain the image size (stride 1, padding 2)
    Columns are concatenated and passed through two fully connected aggregation layers
    """
    def __init__(self):
        super(SB_CNN_6, self).__init__()
        
        self.features = nn.Sequential(
            # conv1 8 filters 5x5
            nn.Conv2d(1, 8, bias=True, kernel_size=5, stride=1, padding=2), # 1x64x64 -> 8x64x64
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8x64x64 -> 8x32x32
            
            # conv2 16 filters 5x5
            nn.Conv2d(8, 16, bias=True, kernel_size=5, stride=1, padding=2), # 8x32x32 -> 16x32x32
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x32x32 -> 16x16x16
        )
        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(2*16*16*16, 2048), # 8192 -> 2048
            nn.ReLU(inplace=True),
            
            #fc2
            nn.Linear(2048, 512), # 2048 -> 512
            nn.ReLU(inplace=True),
            
            # output
            nn.Linear(512, 1), # 512 -> 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        a, b = x
        N = a.size(0)
        
        # 1x64x64 -> 16x16x16
        a = self.features(a)
        b = self.features(b)
        
        # 16x16x16 -> 4096
        a = a.view(N, 16*16*16)
        b = b.view(N, 16*16*16)
        
        # 4096 + 4096 -> 8192
        x = torch.cat((a,b),1)
        
        # 8192 -> 1
        x = self.classifier(x)
        return x


# SB CNN 7
# Closer to VGG models: 3x3 convs, deeper
class SB_CNN_7(_Basemodel):
    r"""
    Features
    Three sets of two convolutions followed by maxpool
    Total of six conv layers, batchnorm, ReLU
    
    Classifier
    Two fully-connected layers, ReLU
    
    Output
    One neuron, logistic regression
    """
    def __init__(self):
        super(SB_CNN_7, self).__init__()
        
        self.features = nn.Sequential(
            # conv1 16 filters 3x3
            nn.Conv2d(1, 16, bias=True, kernel_size=3, stride=1, padding=1), # 1x64x64 -> 16x64x64
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            
            # conv2 16 filters 3x3
            nn.Conv2d(16, 16, bias=True, kernel_size=3, stride=1, padding=1), # 16x64x64 -> 16x64x64
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x64x64 -> 16x32x32
            
            # conv3 32 filters 3x3
            nn.Conv2d(16, 32, bias=True, kernel_size=3, stride=1, padding=1), # 16x32x32 -> 32x32x32
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            
            # conv4 32 filters 3x3
            nn.Conv2d(32, 32, bias=True, kernel_size=3, stride=1, padding=1), # 32x32x32 -> 32x32x32
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32x32 -> 32x16x16
            
            # conv5 64 filters 3x3
            nn.Conv2d(32, 64, bias=True, kernel_size=3, stride=1, padding=1), # 32x16x16 -> 64x16x16
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
            
            # conv6 64 filters 3x3
            nn.Conv2d(64, 64, bias=True, kernel_size=3, stride=1, padding=1), # 64x16x16 -> 64x16x16
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2) # 64x16x16 -> 64x8x8
        )
        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(2*64*8*8, 2048), # 8192 -> 2048
            nn.ReLU(inplace=True),
            
            #fc2
            nn.Linear(2048, 2048), # 2048 -> 2048
            nn.ReLU(inplace=True),
            
            # output
            nn.Linear(2048, 1), # 2048 -> 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        a, b = x
        N = a.size(0)
        
        # 1x64x64 -> 16x16x16
        a = self.features(a)
        b = self.features(b)
        
        # 16x16x16 -> 4096
        a = a.view(N, 64*8*8)
        b = b.view(N, 64*8*8)
        
        # 4096 + 4096 -> 8192
        x = torch.cat((a,b),1)
        
        # 8192 -> 1
        x = self.classifier(x)
        return x


# SB CNN 8
# Model 6 with more filters in both convs
class SB_CNN_8(_Basemodel):
    r"""
    Features
    Two sets of:
    convolution 5x5, batchnorm, ReLU, maxpool
    
    Classifier
    Two fully-connected layers, ReLU
    
    Output
    One neuron, logistic regression
    """
    def __init__(self):
        super(SB_CNN_8, self).__init__()
        
        self.features = nn.Sequential(
            # conv1 32 filters 5x5
            nn.Conv2d(1, 32, bias=True, kernel_size=5, stride=1, padding=2), # 1x64x64 -> 32x64x64
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x64x64 -> 32x32x32
            
            # conv2 64 filters 5x5
            nn.Conv2d(32, 64, bias=True, kernel_size=5, stride=1, padding=2), # 32x32x32 -> 64x32x32
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64x32x32 -> 64x16x16
        )
        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(64*16*16*2, 8192), # 32768 -> 8192
            nn.ReLU(inplace=True),
            
            #fc2
            nn.Linear(8192, 2048), # 8192 -> 2048
            nn.ReLU(inplace=True),
            
            # output
            nn.Linear(2048, 1), # 2048 -> 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        a, b = x
        N = a.size(0)
        
        # 1x64x64 -> 64x16x16
        a = self.features(a)
        b = self.features(b)
        
        # 64x16x16 -> 16384
        a = a.view(N, 64*16*16)
        b = b.view(N, 64*16*16)
        
        # 16384 + 16384 -> 32768
        x = torch.cat((a,b),1)
        
        # 32768 -> 1
        x = self.classifier(x)
        return x
