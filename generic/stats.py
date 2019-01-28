#!/usr/bin/env python

import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt


class Logger(object):
    r""" Keeps a CSV log file
    Constructor arguments:
        columns     variable number of column names
        file        path and filename for log output (default: './log.csv')
    """
    def __init__(self, *columns, file='./log.csv'):
        self.extension = '.csv'
        
        # Convert each item to string and wrap in quotes
        self.columns = ['"{}"'.format(col) for col in columns]
        self.num_cols = len(columns) # Save number of columns
        self.file = file
        
        self.reset()
    
    def reset(self):
        r""" Open empty file, create if it does not exist.
        Write header to the beginning of the file. """
        header = ','.join(self.columns)
        with open(self.file, 'w') as f:
            f.write(header)
            f.write('\n')
    
    def update(self, *data):
        r""" Append data in new line to end of file. """
        assert len(data) == self.num_cols, \
            "Number of data items ({}) does not match number of columns ({})".format(
            len(data), self.num_cols)
        
        line = ','.join(['{}'.format(item) for item in data])
        with open(self.file, 'a') as f:
            f.write(line)
            f.write('\n')


class Plotter(object):
    r"""
    Lazy read: save relevant information at initialization,
               but do not read data until saving the plots
    """
    def __init__(self, file='./log.csv', plots=[]):
        self.file = file
        self.plots = plots
        self.df = None
    
    def read(self):
        if self.df == None:
            self.df = pd.read_csv(self.file)
    
    def save(self):
        r""" Plot specified columns
        Read data with pandas
        """
        self.read()
        for plot in self.plots:
            filename = plot[0]
            kwargs = plot[1]
            ax = self.df.plot(**kwargs)
            if len(plot) > 2: # optional legend
                ax.legend(plot[2])
            fig = ax.get_figure()
            fig.savefig(filename)
