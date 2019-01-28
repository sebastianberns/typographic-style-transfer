#!/usr/bin/env python

from PIL import Image

import numpy as np
import torch
import torchvision as tv


def load_font(fontfile):
    """
    Load all glyphs of a given font
    and return as tensor of size (G x C x W x H)
    26 x 1 x 64 x 64
    """
    
    font_transforms = tv.transforms.Compose([
        tv.transforms.Grayscale(),
        Invert(),
        tv.transforms.ToTensor()
    ])
    
    image = Image.open(fontfile)
    image = font_transforms(image)
    glyphs = torch.split(image, config.SIZE, dim=2)
    return glyphs


def load_font_abc(fontfile):
    """
    Load font
    and set each glyph as single example
    """
    glyphs = load_font(fontfile)
    font = torch.cat(glyphs, dim=0).unsqueeze(1)
    return font


def batch_aabbcc(font):
    """
    Given a font (26 x 1 x 64 x 64)
    split into glyphs
    repeat each glyphs 26 times vertically
    and concatenate
    """
    glyphs = torch.split(font, 1, dim=0)
    batch = torch.zeros((batchsize, 1, config.SIZE, config.SIZE), device=font.device)
    for i, glyph in enumerate(glyphs):
        x = i * config.LETTERS
        y = x + config.LETTERS
        batch[x:y, :, :, :] = glyph.repeat(config.LETTERS, 1, 1, 1)
    return batch


def load_font_abcabc(fontfile):
    """
    Load font
    concatenate glyphs vertically
    and repeat 26 times
    """
    glyphs = load_font(fontfile)
    font = torch.cat(glyphs, dim=0)
    batch = font.repeat(config.LETTERS, 1, 1)
    return batch.unsqueeze(1)


def rand_choice(l):
    options = np.arange(l) # 0-25
    selection = np.random.choice(options, l) # Pick 26 times out of 0-25
    return selection


def rand_glyphs(font):
    return font[rand_choice(font.size(0)), :, :, :]


def savedata(data, step):
    filename = os.path.join(savedir, "{}.npy".format(step))
    np.save(filename, data.detach().cpu().numpy())
