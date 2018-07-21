'''
Created on Jul 21, 2018

@author: loitg
'''
import cv2, os
import numpy as np
from pytesseract import pytesseract
from scipy.misc import imsave
import codecs
import enchant
from utils.common import sauvola, sharpen, firstAnalyse
import re

from algorithm.doubleextractbinarize import pipolarRotateExtractLine, removedot

class UnicodeUtil(object):
    def __init__(self, diacritics_path_csv):
        self.accent_dict = {}
        with open(diacritics_path_csv) as f:
            for line in f:
                temp = line.strip().split(',')
                accent = temp[1].decode('utf8')                
                self.accent_dict[accent] = temp[2:]
    
    def charList(self):
        return list(self.accent_dict.keys())
        

class Validate(object):
    
    def __init__(self):
        self.vdict = enchant.Dict('vi')
        self.date_p = re.compile('([012]?[\d/]|3[01])[-/\.]([012]?[\d/]|3[01])[-/\.]((200|19[5-9/])[\d/])')
        self.accent_dict = UnicodeUtil('/home/loitg/workspace/genline/resource/diacritics2.csv')
        
    def _clean(self, keep_symbols, txt):
        #remove space
        txt = txt.strip()
        txt = txt.replace(u',', u' ')
        txt = re.sub(' +',' ',txt)
        #remove nonsense symbols
        txt = ''.join([c for c in txt if c in keep_symbols])
        return txt
    
    def valCmndDob(self, txt):
        m = self.date_p.match(txt)
        if m is not None:
            return True, m.group(1) + '-' + m.group(2) + '-' + m.group(3)
        else:
            return 0.0, None
        
    def isValidID(self, txt):
        return len(txt) == 9, txt
    
    def isValidAddrCode(self, txt):
        digit = sum(c.isdigit() for c in txt)
        alpha = sum(c.isalpha() for c in txt)
    #     return txt.upper() == txt and 1.0*(digit + alpha)/len(txt) > 0.8
        if len(txt) > 0:
            return 1.0*(digit + alpha)/len(txt), ''
        else:
            return 0.0, ''
        
    def isValidName(self, txt):
        txt = self._clean(self.accent_dict.charList() + [u' '], txt)
        print '---0 ', txt
        hasMeaning = 0; wcount = 0
        for w in txt.split(' '):
            if len(w) == 0: continue
            if self.vdict.check(w):
                hasMeaning += 1
            wcount += 1
        if wcount > 0:
            return 1.0*hasMeaning/wcount, txt
        else:
            return 0.0, ''
        

def isValid(txt):
    pass
    

def loop_ocr(img, input_type='raw', charset=None, lang='eng', lstm=False):
    ### config string from param
    config = ''
    if charset:
        config += 'tessedit_char_whitelist="' + charset + '" '
    config += "-l " + lang + " "
    if lstm:
        config += "--oem 1 "
    else:
        config += "--oem 0 "
    config += "--psm 7 "
    ### To gray-scale
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_grey = img[:,:,...]
        
#     print 'img', img.shape
#     print 'img_grey', img_grey.shape
    ### Loop k
    validator = Validate()
    for k in np.linspace(0.3,0.6,3):
        binline = sauvola(sharpen(img_grey), img_grey.shape[0]/4, k, reverse=True)
        scale = binline.shape[0]/4
        def lambda_extract(o, arg_scale):
            h = o[0].stop - o[0].start
            w = o[1].stop - o[1].start
            return h < scale/2 and w < scale/2
        objects, extracts, _ = firstAnalyse(binline, [lambda_extract])
#         binline = removedot(binline, extracts[0], scale)
        binline = cv2.subtract(binline, extracts[0])
        
#         binline = cv2.resize(binline, None, fx=2.0, fy=2.0)
        binline = np.where(binline > 0.5, np.uint8(0), np.uint8(255))
        txt = pytesseract_ocr(binline, config)
        print k, '@@--@@', txt
#         print 'dob', validator.valCmndDob(txt)
#         print 'id', validator.isValidID(txt)
#         print 'code', validator.isValidAddrCode(txt)
#         print 'name', validator.isValidName(txt)
#         cv2.imshow('img', binline)
#         cv2.waitKey(-1)
            
    return txt
            

def pytesseract_ocr(img, config=''):
    """Runs Tesseract on a given image. Writes an intermediate tempfile and then runs the tesseract command on the image.

    This is a simplified modification of image_to_string from PyTesseract, which is adapted to SKImage rather than PIL.

    In principle we could have reimplemented it just as well - there are some apparent bugs in PyTesseract (e.g. it
    may lose the NamedTemporaryFile due to its auto-delete behaviour).

    :param mrz_mode: when this is True (default) the tesseract is configured to recognize MRZs rather than arbitrary texts.
    """
    input_file_name = '%s.bmp' % pytesseract.tempnam()
    output_file_name_base = '%s' % pytesseract.tempnam()
    output_file_name = "%s.txt" % output_file_name_base
    try:
        imsave(input_file_name, img)
        status, error_string = pytesseract.run_tesseract(input_file_name,
                                             output_file_name_base,
                                             lang=None,
                                             boxes=False,
                                             config=config)
        if status:
            errors = pytesseract.get_errors(error_string)
            raise pytesseract.TesseractError(status, errors)
        if 'vie' in config:
            f = codecs.open(output_file_name, encoding='utf-8')
        else:
            f = open(output_file_name)
        try:
            return f.read().strip()
        finally:
            f.close()
    finally:
        pytesseract.cleanup(input_file_name)
        pytesseract.cleanup(output_file_name)
        
if __name__ == '__main__':
    linedir = '/home/loitg/workspace/clocr/temp/0/'
    for linepath in os.listdir(linedir):
        oriLine = cv2.imread(linedir+linepath)
        extLine = pipolarRotateExtractLine(oriLine, 0.5)
        txt =  loop_ocr(extLine, input_type='raw', lang='vie', lstm=True)
#         txt =  loop_ocr(extLine, input_type='raw', charset='0123456789', lstm=False)
        print type(txt)
        print txt
        
    