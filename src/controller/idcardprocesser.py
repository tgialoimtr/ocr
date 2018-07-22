# -*- coding: utf-8 -*-
'''
Created on Jul 10, 2018

@author: loitg
'''
import os
import cv2
import json
from extras.weinman.interface.linepredictor import BatchLinePredictor
from utils.common import createLogger, args
from cmnd.template import StaticTemplate
from algorithm.looptess import loop_ocr, UnicodeUtil
from algorithm.doubleextractbinarize import pipolarRotateExtractLine
import editdistance
import re, enchant

class Validate(object):
    
    def __init__(self):
        self.vdict = enchant.Dict('vi')
        self.date_p = re.compile('([012]?[\d/]|3[01])[-/\. ]?([012]?[\d/]|3[01])[-/\. ]?[ ]?((200|19[5-9/])[\d/])')
        self.accent_dict = UnicodeUtil('/home/loitg/workspace/genline/resource/diacritics2.csv')
    
    def _correctWord(self, word):
        if word == u'TP': return word
        wordTI = word.replace(u'T',u'I')
        if self.vdict.check(word):
            return word
        elif self.vdict.check(wordTI):
            return wordTI
        else:
            nearWord = self.vdict.suggest(word)
            if len(nearWord) > 0:
                nearWord = nearWord[0]
                if 1.0*editdistance.eval(nearWord, word)/len(nearWord) < 0.34:
                    return nearWord
                else:
                    return None
            else:
                return None
    
    def _clean(self, keep_symbols, txt):
        #remove space
        txt = txt.replace(u',', u' ')
        txt = txt.replace(u'—',u'-').replace(u'²',u'2').replace(u'×',u'.').replace(u'.', u' ')\
                .replace(u',',u' ').replace(u'¥', u'Y').replace(u'ï',u'i').replace(u'º',u'o').replace(u'—319',u'-19')
        txt = re.sub(' +',' ',txt)
        #remove nonsense symbols
        txt = ''.join([c for c in txt if c in keep_symbols])
        return txt.strip()
    
    def correctID(self, txt):
        txt = self._clean(list(u'0123456789'), txt)
        if len(txt) == 9 or len(txt) == 12:
            return 1.0, txt
        else:
            return 0.0, None
        
    def correctDOB(self, txt):
        txt = self._clean(list(u' 0123456789-.'), txt)
        m = self.date_p.match(txt)
        if m is not None:
            return 1.0, m.group(1) + '-' + m.group(2) + '-' + m.group(3)
        else:
            return 0.0, None
        
    def correctGender(self, txt):
        pass

    def correctName(self, txt):
        txt = self._clean(self.accent_dict.charList() + list(u' ,-'), txt)
        meaningfulContinuous = 0; wcount = 0; rs = u''
        for w in txt.split(' '):
            if len(w) == 0: continue
            neww = self._correctWord(w)
            if neww:
                meaningfulContinuous += 1
                rs += neww + u' '
            else:
                meaningfulContinuous = 0
                rs = u''
            wcount += 1
        if wcount > 0:
            return 1.0*meaningfulContinuous/wcount, rs
        else:
            return 0.0, ''

def validateLine(line):
    newwidth = 50.0/line.shape[0]*line.shape[1]
    if newwidth > 1500:
        return None
    else:
        return cv2.resize(line, (int(newwidth), 50))
    
class Cmnd9Processer(object):


    def __init__(self, bzid, logger=None):
        if logger is None: logger = createLogger('bzid')
        self.logger = logger
        self.cmnd9_tmpl = StaticTemplate.createFrom('/home/loitg/workspace/clocr/temp/bzcmnd9/template/1cmnd9.xml',
                                                    '/home/loitg/workspace/clocr/temp/bzcmnd9/template/1cmnd9.jpg',
                                                    'CMND Cu - 9 So')
        self.cancuoc_tmpl = StaticTemplate.createFrom('/home/loitg/workspace/clocr/temp/bzcmnd9/template/0cancuoc.xml',
                                                    '/home/loitg/workspace/clocr/temp/bzcmnd9/template/0cancuoc.jpg',
                                                    'Can Cuoc Cong Dan')
        self.cmnd12_tmpl = StaticTemplate.createFrom('/home/loitg/workspace/clocr/temp/bzcmnd9/template/cmnd12.xml',
                                                    '/home/loitg/workspace/clocr/temp/bzcmnd9/template/cmnd12.jpg',
                                                    'CMND Moi - 12 So')
        self.list_tmpl = [self.cmnd9_tmpl, self.cancuoc_tmpl, self.cmnd12_tmpl]   
    
    def init(self):
        pass
    
    
    ### This is NOT multi-thread safe
    def process1(self, filepath):
        imgwhole = cv2.imread(filepath)
        probs_lines_tmpl = [(tmpl.find(imgwhole) + (tmpl,)) for tmpl in self.list_tmpl]
        probs_lines_tmpl.sort(reverse=True)
        if probs_lines_tmpl[0][0] < 0.5:
            return {}
        lines = probs_lines_tmpl[0][1]
        rs = {}
        corrector = Validate()
        rs['type'] = probs_lines_tmpl[0][2].desc
        img = validateLine(pipolarRotateExtractLine(lines['id'].img, 0.5))
        rs['idNumber'] = loop_ocr(img, input_type='raw', lang='vie', lstm=True, corrector=corrector.correctID) \
                        if img is not None else u''
        img = validateLine(pipolarRotateExtractLine(lines['ntns'].img, 0.5))
        rs['dateOfBirth'] = loop_ocr(img, input_type='raw', lang='vie', lstm=True, corrector=corrector.correctDOB)\
                        if img is not None else u''
#         rs['Gender']
#         rs['Dantoc']
        img = validateLine(pipolarRotateExtractLine(lines['quequan1'].img, 0.5))
        quequan1 = loop_ocr(img, input_type='raw', lang='vie', lstm=True, corrector=corrector.correctName)\
                        if img is not None else u''
        img = validateLine(pipolarRotateExtractLine(lines['quequan2'].img, 0.5))
        quequan2 = loop_ocr(img, input_type='raw', lang='vie', lstm=True, corrector=corrector.correctName)\
                        if img is not None else u''
        rs['NguyenQuan'] = quequan1 + quequan2
        img = validateLine(pipolarRotateExtractLine(lines['hoten1'].img, 0.5))
        hoten1 = loop_ocr(img, input_type='raw', lang='vie', lstm=True, corrector=corrector.correctName)\
                        if img is not None else u''
        img = validateLine(pipolarRotateExtractLine(lines['hoten2'].img, 0.5))
        hoten2 = loop_ocr(img, input_type='raw', lang='vie', lstm=True, corrector=corrector.correctName)\
                        if img is not None else u''
        
        temp = editdistance.eval(hoten1, hoten2)
        if temp < 4: hoten1 = u''
        rs['fullName'] = hoten1 + hoten2



        print probs_lines_tmpl[0][0], probs_lines_tmpl[0][2].desc
        print rs
        cv2.imshow('fadsfs', imgwhole)
        cv2.waitKey(-1)
        return json.dumps(rs) 



    ### This is multi-thread safe
    def read(self, filepath):
        return self.process1(filepath)
    
    
if __name__ == '__main__':
    p = Cmnd9Processer('cmnd9')
    p.init()
    pathp = '/home/loitg/workspace/clocr/temp/1/'
    for fn in os.listdir(pathp):
        a = p.read(pathp + fn)


 
