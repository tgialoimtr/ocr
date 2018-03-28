'''
Created on Mar 15, 2018

@author: loitg
'''

    
class Line(object):
    
    def __init__(self, bounds = None, img = None, shape = None):
        self.bounds = bounds
        self.text = ''
        self.img = img
        if self.img is not None:
            self.shape = self.img.shape
        else:
            self.shape = None