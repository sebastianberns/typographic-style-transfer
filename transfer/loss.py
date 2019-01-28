#!/usr/bin/env python

import torch


''' Grey Mean/Ratio Loss '''

def greymask(input):
    """
    Compute a mask for grey scale pixel values
    0.  where pixel value is 0 or 1
    1.  otherwise
    """
    one = torch.tensor(1., device=input.device)
    zero = torch.tensor(0., device=input.device)
    black = torch.where(input == 0, one, zero)
    white = torch.where(input == 1, one, zero)
    return torch.ones_like(input) - black - white


def greymean(input):
    """
    Get grey scale mask
    Multiply input by mask and sum -> all 0 and 1 pixel values will be ignored
    Divide by sum of mask -> number of grey pixels in input
    """
    mask = greymask(input)
    return torch.sum(torch.sum(input*mask, 3), 2) / torch.sum(torch.sum(mask, 3), 2)

def greymean_loss(input, target):
    """
    Compute square distance between mean grey value of input and target
    """
    return torch.mean((greymean(input) - target) ** 2)


def greyratio(input):
    """
    Sum grey scale mask
    Divide by total number of pixels
    """
    grey = torch.sum(torch.sum(greymask(input), 3), 2)
    total = input.view(config.LETTERS,-1).size(1)
    return grey/total

def greyratio_loss(input, target):
    """
    Compute square distance of input's grey ratio from target
    """
    return torch.mean((greyratio(input) - target) ** 2)


''' Total Variation Loss '''

def tvloss_abs(input):
    ''' Absolute distance '''
    horizontal = torch.sum(torch.sum( torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:] ), 3), 2)
    vertical   = torch.sum(torch.sum( torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :] ), 3), 2)
    return torch.mean(horizontal + vertical)

def tvloss_sq(input):
    ''' Square distance '''
    horizontal = torch.sum(torch.sum( (input[:, :, :, :-1] - input[:, :, :, 1:])**2, 3), 2)
    vertical   = torch.sum(torch.sum( (input[:, :, :-1, :] - input[:, :, 1:, :])**2, 3), 2)
    return torch.mean(horizontal + vertical)
