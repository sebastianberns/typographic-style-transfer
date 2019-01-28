#!/usr/bin/env python

import copy
import os

import torch
import torch.nn as nn


class Supermodel(nn.Module):
    r"""
    General wrapper class for different network architectures
    
    Params:
        models      List of available models (from imported style.models.list)
        modelname   Class name of model (as defined in models package)
    
    Attributes:
        best        Best performance (epoch, accuracy, avg loss)
    
    Methods: (apart from nn.Module inheritance)
        no_grad     Set requires_grad flag to False for all model parameters
        save        Save network parameters
        load        Load network parameters
    """
    def __init__(self, models, modelname, device=None):
        super(Supermodel, self).__init__()
        
        if not modelname:
            raise ValueError("Define network model (--model=NAME)")
        
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.models = models
        if modelname not in self.models: # check if available
            raise ValueError("Model not found '{}'".format(modelname))
        
        self.model = self.models[modelname]().to(device) # call model constructor
        self.name = self.model._get_name()
        self.extension = '.pth'
        
        # STATS (epoch, error, loss)
        self._best_default = (None, 1., 0.)
        self._best = self._best_default
    
    # ATTRIBUTES
    
    r""" Set, get and reset information about best epoch """
    @property
    def best(self):
        return self._best
    @best.setter
    def best(self, *input):
        self._best = tuple(*input)
    @best.deleter
    def best(self):
        self._best = self._best_default
    
    # METHODS
    
    def no_grad(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        return self.model.eval()
    
    def forward(self, *input):
        return self.model.forward(*input)
    
    def _rmsaved(self, path='./'):
        r""" Remove previously saved model state dicts """
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath) \
            and filename.startswith(self.name) \
            and filename.endswith(self.extension):
                os.remove(filepath)
    
    def save(self, epoch, path='./', rmprev=True):
        r""" Save model state dict """
        if rmprev:
            self._rmsaved(path)
        filename = '{}_epoch{}'.format(self.name, epoch)
        for device in ['cuda','cpu']:
            torch.save(copy.deepcopy(self.model).to(device).state_dict(),
                os.path.join(path, '{}_{}{}'.format(filename, device, self.extension)))
        return filename
    
    def load(self, path='./'):
        r""" Load model state dict """
        self.model.load_state_dict(torch.load(path))
