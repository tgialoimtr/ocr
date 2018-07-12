'''
Created on Mar 8, 2018

@author: loitg
'''

from pylab import *
import Image
import pytesseract
import lineest
from utils.common import sauvola, ASHOW
import cv2

class TessLineRecognizer(object):
    '''
    classdocs
    '''
    
    LSTM = 'LSTM'
    LEGACY = 'LEGACY'
    
    def __init__(self):
        pass
        
        
    def read(self, imgline, mode=LSTM):
        '''
        This is messy now.
        Maybe not use anymore
        If use, debug and fix this
        '''
        print 'READ'
        grey = cv2.cvtColor(imgline, cv2.COLOR_BGR2GRAY)
        print '0--' + str(np.amax(grey)) + '--' + str(np.amin(grey))
        binline = sauvola(grey, w=int(imgline.shape[0]*3.0/4.0))
        if np.amax(binline) == np.amin(binline): return ''
        print '1--' + str(np.amax(binline)) + '--' + str(np.amin(binline))
        binline = (1.0-binline) # black to white text
        ASHOW('2', binline,waitKey=True)
        le = lineest.CenterNormalizer(binline.shape[0]) # white text
        print '3'
        binline = binline.astype(float)
        print '4--' + str(np.amax(binline)) + '--' + str(np.amin(binline))
        le.measure(binline)
        print '5'
        binline = le.normalize(grey)
        print '6'
        
        
        print '-----------------------'
        pilimg = Image.fromarray((binline*255).astype(uint8))
        if mode == self.LSTM:
            pred = pytesseract.image_to_string(pilimg,lang='eng', config='--oem 0 --psm 7')
            print '00', pred
        else:
            pred = pytesseract.image_to_string(pilimg,lang='eng', config='--oem 1 --psm 7')
            print '11', pred
        
        return pred