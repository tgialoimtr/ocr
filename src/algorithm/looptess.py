'''
Created on Jul 21, 2018

@author: loitg
'''
import cv2, os
import numpy as np
from pytesseract import pytesseract
from scipy.misc import imsave
import codecs
from utils.common import sauvola, sharpen, firstAnalyse, TEMPORARY_PATH

from algorithm.doubleextractbinarize import pipolarRotateExtractLine, removedot

class VNCharInfo(object):
    def __init__(self, base, accent0, accent1):
        self.base = base
        self.accent0 = accent0
        self.accent1 = accent1
        
class UnicodeUtil(object):
    def __init__(self, diacritics_path_csv):
        self.accent_dict = {}
        with open(diacritics_path_csv) as f:
            for line in f:
                temp = line.strip().split(',')
                accent = temp[1].decode('utf8')                
                self.accent_dict[accent] = VNCharInfo(temp[2], temp[3], temp[4])
    
    def charList(self):
        return list(self.accent_dict.keys())
    
    def at(self, ch):
        if ch in self.accent_dict:
            return self.accent_dict[ch]
        else:
            return None
    
def loop_ocr(img, input_type='raw', charset=None, lang='eng', lstm=False, corrector=None):
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
    ### Loop k
    rs = []
    for k in np.linspace(0.35,0.55,2):
        binline = sauvola(sharpen(img_grey), img_grey.shape[0]/4, k, reverse=True)
        scale = binline.shape[0]/4
        def lambda_extract(o, arg_scale):
            h = o[0].stop - o[0].start
            w = o[1].stop - o[1].start
            return (h < scale/2 and w < scale/2) or (1.0*w/h > 6)
        objects, extracts, _ = firstAnalyse(binline, [lambda_extract])
#         binline = removedot(binline, extracts[0], scale)
        binline = cv2.subtract(binline, extracts[0])
        binline = np.where(binline > 0.5, np.uint8(0), np.uint8(255))
        txt = pytesseract_ocr(binline, config)
#         print k, '@@--@@', txt
        if corrector:
            rs.append(corrector(txt))
        else:
            return txt
    rs.sort(reverse=True)
    print '---', rs[0][1]
    return rs[0][1]
            

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
    linedir = TEMPORARY_PATH + '0/'
    for linepath in os.listdir(linedir):
        oriLine = cv2.imread(linedir+linepath)
        extLine = pipolarRotateExtractLine(oriLine, 0.5)
        txt =  loop_ocr(extLine, input_type='raw', lang='vie', lstm=True)
        print txt
        
    