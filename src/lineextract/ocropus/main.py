# -*- coding: utf-8 -*-
'''
Created on Oct 16, 2017

@author: loitg
'''
import os,sys
import cv2
import numpy as np
from utils.common import ASHOW
from lineextract.ocropus.line_extractor import LinesExtractor as OcrLineExtractor

partall_path = '/home/loitg/Downloads/part2/'
output_path = '/home/loitg/workspace/receipttest/rescources/db/'

if __name__ == '__main__':
    cmnd_path = '/home/loitg/Downloads/cmnd_data/'
    hoadon_path = '/home/loitg/workspace/python/python/img/'
    imgpath = hoadon_path
    extor = OcrLineExtractor()
    for fn in os.listdir(imgpath):
        print fn
        if fn[-3:].upper() not in ['PEG','JPG']: continue
        img = cv2.imread(imgpath + fn)
        ASHOW('gg', img, waitKey=True)
        extor.extractLines(imgpath + fn)
            