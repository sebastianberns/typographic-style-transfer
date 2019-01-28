#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate, _DataLoaderIter



class TrainLoader(torch.utils.data.DataLoader):
    r"""
    Get a batch of M fonts from default DataLoader,
      where M=N/2 and N is the batch size
    Generate M positive and M negative examples,
      one for each font in the batch
    Each sample consists of two separate letter images
    """
    def __init__(self, dataset, config, batch_size=4, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        
        self.config = config
        
        assert batch_size >= 4, "A batch has to have at least two different fonts, and thus size >= 4"
        assert batch_size % 2 == 0, "The batch size has to me a multiple of 2"
        
        self.actual_batch_size = batch_size # overwriting 'batch_size' is blocked by superclass
        
        # call DataLoader constructor
        super(TrainLoader, self).__init__(dataset, batch_size=int(self.actual_batch_size/2), shuffle=shuffle,
              sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
              collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
              timeout=timeout, worker_init_fn=worker_init_fn)

    def random_chars(self):
        # Select two random letters
        a,b = np.random.choice(self.config.ALPHABET, size=2)
        if a==b: # if a,b identical
            a,b = self.random_chars()
        return a,b
    
    def random_font(self, M, i):
        # Select a random font, different from i
        j = np.random.choice(np.arange(M))
        if j==i: # if fonts are identical
            j = self.random_font(M, i)
        return j
    
    def glyph(self, font, i):
        i = i.upper()
        p = self.config.LETTERPOS[i]
        q = p + self.config.SIZE
        glyph = font[:,:,p:q] # (C x H x W)
        return glyph
    
    def positive_sample(self, fonts, i):
        font = fonts[i,:,:,:]
        a,b = self.random_chars()
        
        A = self.glyph(font, a)
        B = self.glyph(font, b)
        return A,B
    
    def negative_sample(self, fonts, i):
        M = fonts.size(0)
        j = self.random_font(M, i)
        a,b = self.random_chars()
        
        A = self.glyph(fonts[i,:,:,:], a)
        B = self.glyph(fonts[j,:,:,:], b)
        return A,B
    
    def unison_shuffled_copies(self, a, b, c):
        # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison#4602224
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]
    
    def __iter__(self):
        self.iterator = _DataLoaderIter(self)
        return self
    
    def __next__(self):
        r"""
        Call DataLoader iterator
        Fonts batch has shape (M x C x H x W)
          M  number of fonts
          C  1 grey channel
          H  64 px height
          W  1664 px width of 26 letters
        Generate double the amount of given fonts in a batch
          1 positive and
          1 negative sample per font
        Return in random order
        """
        fonts = self.iterator.__next__()
        M = fonts.size(0) # number of fonts
        
        # (N x C x H x W)
        feats_a = np.zeros((M*2, 1, self.config.SIZE, self.config.SIZE), dtype=np.float32)
        feats_b = np.zeros((M*2, 1, self.config.SIZE, self.config.SIZE), dtype=np.float32)
        targets = np.zeros((M*2, 1), dtype=np.float32)
        
        for i in range(M):
            # POSITIVE SAMPLE
            feats_a[i,:,:,:], feats_b[i,:,:,:] = self.positive_sample(fonts, i)
            targets[i] = 1.0 # true
            
            # NEGATIVE SAMPLE
            feats_a[i+M,:,:,:], feats_b[i+M,:,:,:] = self.negative_sample(fonts, i)
            targets[i+M] = 0.0 # false
        
        # shuffle
        feats_a, feats_b, targets = self.unison_shuffled_copies(feats_a, feats_b, targets)
        
        # x:(a,b), y
        return (torch.from_numpy(feats_a), torch.from_numpy(feats_b)), torch.from_numpy(targets)



class EvalLoader(torch.utils.data.DataLoader):
    r"""
    Load pre-generated PNG images
    Split into two letter images
    Return together with targets
    """
    def __init__(self, dataset, config, batch_size=2, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        
        self.config = config
        
        # call DataLoader costructor
        super(EvalLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
              sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
              collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
              timeout=timeout, worker_init_fn=worker_init_fn)
    
    def __iter__(self):
        self.iterator = _DataLoaderIter(self)
        return self
    
    def __next__(self):
        data, target = self.iterator.__next__() # call DataLoader iterator
        a, b = data.split(self.config.SIZE, dim=3)
        return (a, b), target
