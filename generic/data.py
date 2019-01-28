#!/usr/bin/env python

from PIL import Image, ImageOps
import os
import torch
import torchvision as tv



class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """
    
    def invert(self, img):
        r"""Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        if not tv.transforms.functional._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            inv = ImageOps.invert(rgb)
            r, g, b = inv.split()
            inv = Image.merge('RGBA', (r, g, b, a))
        elif img.mode == 'LA':
            l, a = img.split()
            l = ImageOps.invert(l)
            inv = Image.merge('LA', (l, a))
        else:
            inv = ImageOps.invert(img)
        return inv

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        return self.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'



# https://gist.github.com/ikhlestov/0f174783eb8b37a77ab34c07f21ccd6a
class MCGAN_dataset(torch.utils.data.Dataset):
    r"""
    Read font files in data directory ('datadir')
    Provide complete font (uppercase letters A-Z) by index
    """
    def __init__(self, datadir, config, transform=None):
        if not os.path.exists(datadir):
            raise FileNotFoundError(datadir)
        
        self.datadir = datadir            # directory of data
        self.files = os.listdir(datadir)  # list of files in datadir
        
        self.config = config
        
        self.transform = transform
    
    def loader(self, filename):
        # Read greyscale image (1664x64) of range [0,255]
        # Return as 1-d vector of length 106496 (64*64*26)
        I = Image.open( os.path.join(self.datadir, filename) )

        assert I.size == (self.config.SIZE*26, self.config.SIZE), \
          "Sample '{}' has unexpected size: {}".format(filename, I.size)
        
        return I
    
    def __getitem__(self, index):
        I = self.loader( self.files[index] )
        
        if self.transform is not None:
            I = self.transform(I)
        
        return I
    
    def __len__(self):
        return len(self.files)
