'''
Created on Jun 29, 2018

@author: loitg
'''
import cv2
import sys, os
from cmnd.template import StaticTemplate

class CMNDPredictor(object):
    
    def __init__(self):
        self.cancuoc_tmpl = StaticTemplate.createFrom('/home/loitg/Downloads/cmnd_data/template/1cmnd9.xml')
        
    
    def abc(self, imgpath):
        img = cv2.imread(imgpath)
        prob, lines = self.cancuoc_tmpl.find(img)
        if prob > 0.5:
            print prob
            for k, imgline in lines.iteritems():
                print k
                cv2.imshow('line', imgline.img)
                cv2.waitKey(-1)
        


if __name__ == '__main__':
    sys.argv = ['main.py','/home/loitg/Downloads/cmnd_data/6.jpg','/home/loitg/temp.txt']
    cmnd_path = '/home/loitg/Downloads/cmnd_data/realcapture/'
    p = CMNDPredictor()
    for fn in os.listdir(cmnd_path):
        if fn[-3:].upper() not in ['PEG','JPG','PNG']: continue
        print fn
        p.abc(cmnd_path + fn)