#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _Basemodel
#import config # import together with style.models


# Baluja network 1
class Baluja_1(_Basemodel):
    # Separate tower for both inputs a,b:
    #   1 fully connected
    # Aggregation layers:
    #   2 fully connected
    def __init__(self):
        super(Baluja_1, self).__init__()
        
        self.input_a_lin = nn.Linear(config.SIZE**2, 128)
        self.input_b_lin = nn.Linear(config.SIZE**2, 128)
        
        self.aggr1_lin   = nn.Linear(256, 512)
        self.aggr2_lin   = nn.Linear(512, 512)
        
        self.output_lin  = nn.Linear(512, 1)
    
    def forward(self, x):
        a, b = x
        N = a.size(0) # batch size
        a = a.view(N,-1)
        b = b.view(N,-1)
        
        a = F.relu(self.input_a_lin(a))
        b = F.relu(self.input_b_lin(b))
        
        x = torch.cat((a,b),1) # concatenate a,b along horizontal axis
        x = F.relu(self.aggr1_lin(x))
        x = F.relu(self.aggr2_lin(x))
        
        x = F.sigmoid(self.output_lin(x))
        return x


# Baluja network 2
class Baluja_2(_Basemodel):
    # Separate tower for both inputs a,b:
    #   3 fully connected
    # Aggregation layers:
    #   2 fully connected
    def __init__(self):
        super(Baluja_2, self).__init__()
        
        self.input_a_lin1 = nn.Linear(config.SIZE**2, 128)
        self.input_a_lin2 = nn.Linear(128, 128)
        self.input_a_lin3 = nn.Linear(128, 128)
        
        self.input_b_lin1 = nn.Linear(config.SIZE**2, 128)
        self.input_b_lin2 = nn.Linear(128, 128)
        self.input_b_lin3 = nn.Linear(128, 128)
        
        self.aggr1_lin   = nn.Linear(256, 512)
        self.aggr2_lin   = nn.Linear(512, 512)
        self.output_lin  = nn.Linear(512, 1)
    
    def forward(self, x):
        a, b = x
        N = a.size(0) # batch size
        a = a.view(N,-1)
        b = b.view(N,-1)
        
        a = F.relu(self.input_a_lin1(a))
        a = F.relu(self.input_a_lin2(a))
        a = F.relu(self.input_a_lin3(a))
        
        b = F.relu(self.input_b_lin1(b))
        b = F.relu(self.input_b_lin2(b))
        b = F.relu(self.input_b_lin3(b))
        
        x = torch.cat((a,b),1) # concatenate a,b along horizontal axis
        x = F.relu(self.aggr1_lin(x))
        x = F.relu(self.aggr2_lin(x))
        
        x = F.sigmoid(self.output_lin(x))
        return x


# Baluja network 2
# shared weights across towers
class Baluja_2_B(_Basemodel):
    # Separate tower for both inputs a,b:
    #   3 fully connected
    # Aggregation layers:
    #   2 fully connected
    def __init__(self):
        super(Baluja_2_B, self).__init__()
        
        # towers, shared
        self.input_lin1 = nn.Linear(config.SIZE**2, 128)
        self.input_lin2 = nn.Linear(128, 128)
        self.input_lin3 = nn.Linear(128, 128)
        
        # aggregation
        self.aggr1_lin   = nn.Linear(256, 512)
        self.aggr2_lin   = nn.Linear(512, 512)
        
        # output
        self.output_lin  = nn.Linear(512, 1)
    
    def forward(self, x):
        a, b = x
        N = a.size(0) # batch size
        a = a.view(N,-1)
        b = b.view(N,-1)
        
        a = F.relu(self.input_lin1(a))
        b = F.relu(self.input_lin1(b))
        
        a = F.relu(self.input_lin2(a))
        b = F.relu(self.input_lin2(b))
        
        a = F.relu(self.input_lin3(a))
        b = F.relu(self.input_lin3(b))
        
        x = torch.cat((a,b),1) # concatenate a,b along horizontal axis
        x = F.relu(self.aggr1_lin(x))
        x = F.relu(self.aggr2_lin(x))
        
        x = F.sigmoid(self.output_lin(x))
        return x


# Baluja network 5
# WIP
class Baluja_5(_Basemodel):
    # Separate tower for both inputs a,b:
    #   2 convolutional paths, 2 deep
    #   (3x3 -> 3x3, 4x4 -> 3x3)
    # Aggregation layers:
    #   2 fully connected (100, 100)
    def __init__(self):
        super(Baluja_5, self).__init__()
        
        # A
        # first path
        self.input_a_conv1_1 = nn.Conv2d( 1, 10, kernel_size=3) # 3x3
        self.input_a_conv1_2 = nn.Conv2d(10, 20, kernel_size=3) # 3x3
        # second path
        self.input_a_conv2_1 = nn.Conv2d( 1, 10, kernel_size=4) # 4x4
        self.input_a_conv2_2 = nn.Conv2d(10, 20, kernel_size=3) # 3x3
        
        # B
        # first path
        self.input_b_conv1_1 = nn.Conv2d( 1, 10, kernel_size=3) # 3x3
        self.input_b_conv1_2 = nn.Conv2d(10, 20, kernel_size=3) # 3x3
        # second path
        self.input_b_conv2_1 = nn.Conv2d( 1, 10, kernel_size=4) # 4x4
        self.input_b_conv2_2 = nn.Conv2d(10, 20, kernel_size=3) # 3x3
        
        # aggregation
        self.aggr1_lin   = nn.Linear(100, 100)
        self.aggr2_lin   = nn.Linear(100, 100)
        
        self.output_lin  = nn.Linear(100, 1)
    
    def forward(self, x):
        a, b = x
        a_1 = F.relu(self.input_a_conv1_1(a))
        a_1 = F.relu(self.input_a_conv1_2(a_1))
