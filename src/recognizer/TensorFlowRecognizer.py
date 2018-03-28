'''
Created on Mar 15, 2018

@author: loitg
'''

import cv2
import random

class TensorFlowRecognizer(object):
    '''
    classdocs
    '''
    
    def __init__(self):
        pass
        
    def read(self, imgline):
        idd = str(random.randint(1,999999999))
        cv2.imwrite('/tmp/cmndlines/' + 'DOB' + idd + '.jpg', imgline)
        
        
if __name__ == '__main__':
    pass