#!/usr/bin/env python

import importlib
import os
import pkgutil

'''
https://codereview.stackexchange.com/questions/70268/list-all-classes-in-a-package-directory/70280#70280
1. Import all files in directoy as modules
2. Generate dict (class name, class object) of all subclasses based on _Basemodel
Thus, any newly created class that inherits from _Basemodel will be automatically added to the list
'''
for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
    importlib.import_module('.' + name, __package__)
index = {cls.__name__: cls for cls in base._Basemodel.__subclasses__()}

__all__ = list(index.keys())
