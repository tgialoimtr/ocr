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
from algorithm.looptess import loop_ocr
from algotithm.doubleextractbinarize import pipolarRotateExtractLine

    
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
        img = cv2.imread(filepath)
        probs_lines_tmpl = [(tmpl.find(img) + (tmpl,)) for tmpl in self.list_tmpl]
        probs_lines_tmpl.sort(reverse=True)
        if probs_lines_tmpl[0][0] < 0.5:
            return {}
        lines = probs_lines_tmpl[0][1]
        rs = {}
        rs['type'] = probs_lines_tmpl[0][2].desc
        rs['idNumber'] = loop_ocr(pipolarRotateExtractLine(lines['id'], 0.5), 
                                  input_type='raw', charset='0123456789', lstm=False)
        rs['dateOfBirth'] = loop_ocr(pipolarRotateExtractLine(lines['ntns'], 0.5), 
                                     input_type='raw', charset='0123456789-/', lstm=True)
        rs['Gender'] = loop_ocr(pipolarRotateExtractLine(lines['ntns'], 0.5), 
                                input_type='raw', charset='0123456789-/', lstm=True)
        rs['Dantoc']
        quequan1 = loop_ocr(pipolarRotateExtractLine(lines['quequan1'], 0.5), 
                                input_type='raw', lang='vie', lstm=True)
        quequan2 = loop_ocr(pipolarRotateExtractLine(lines['quequan2'], 0.5), 
                                input_type='raw', lang='vie', lstm=True)
        rs['NguyenQuan'] = quequan1 + quequan2
        hoten1 = loop_ocr(pipolarRotateExtractLine(lines['hoten1'], 0.5), 
                                input_type='raw', lang='vie', lstm=True)
        hoten2 = loop_ocr(pipolarRotateExtractLine(lines['hoten2'], 0.5), 
                                input_type='raw', lang='vie', lstm=True)
        rs['fullName'] = hoten1 + hoten2

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
        print a


 
