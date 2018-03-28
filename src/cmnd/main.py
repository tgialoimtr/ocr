'''
Created on Oct 31, 2017

@author: loitg
'''
import os, sys
from recognizer.TensorFlowRecognizer import TensorFlowRecognizer
# Config PYTHONPATH and template folder relatively according to current file location
project_dir = os.path.dirname(__file__) + '/../'
template_path = project_dir + '/template/'
sys.path.insert(0, project_dir)

import cv2
import datetime, time
from cmnd_info import CMND
from utils.common import estimate_skew_angle, args
from numpy import linspace
import codecs
from multiprocessing import Pool


# Calculate blurness of image using Laplacian operator
def calcBlur(img):
    if len(img.shape) > 1:
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        temp = img
    return cv2.Laplacian(temp, cv2.CV_64F).var()

def fff(c):
    return c[0].findTextRegion(c[1]), c[0]

class CMNDPredictor:  
    template_path = args.template_path
    cmnd12 = CMND('CMND moi - 12 so', template_path + 'cmnd12_tmp.tiff', template_path + 'cmnd12_mask.tiff', template_path + 'cmnd12_tmpname.tiff')
#     cmnd12.addLineDesc({'idNumber':(32,67, 65,306), 'dateOfBirth':(147,165), 'fullName':(62,109,15,334), 'Gender':(165,186), 'Dantoc':(165,186), 'NguyenQuan':(186,226)})
    cancuoc = CMND('Can Cuoc Cong Dan', template_path + 'cancuoc_tmp.tiff', template_path + 'cancuoc_mask.tiff', template_path + 'cancuoc_tmpname.tiff')
    cancuoc.addLineDesc({'idNumber':(40,70), 'dateOfBirth':(120,145), 'fullName':(70,120), 'Gender':(145,175), 'Dantoc':(145,175), 'NguyenQuan':(175,217)})
    cmnd9 = CMND('CMND cu - 9 so', template_path + 'cmnd9_tmp.tiff', template_path + 'cmnd9_mask.tiff', template_path + 'cmnd9_tmpname.tiff')
    cmnd9.addLineDesc({'idNumber':(22,61,83,310,0),'dateOfBirth':(108,143,80,310), 'fullName':(58,87,55,310,0), 'Gender':(0,0), 'Dantoc':(150,207), 'NguyenQuan':(150,207)})
    gplxmoi = CMND('Giay Phep Lai Xe moi', template_path + 'gplxmoi_tmp.tiff', template_path + 'gplxmoi_mask.tiff', template_path + 'gplxmoi_tmpname.tiff')
    gplxmoi.addLineDesc({'idNumber':(35,53), 'dateOfBirth':(81,94), 'fullName':(58,80), 'Gender':(0,0), 'Dantoc':(103,125), 'NguyenQuan':(0,0)})
    gplxcu = CMND('Giay Phep Lai Xe cu', template_path + 'gplxcu_tmp.tiff', template_path + 'gplxcu_mask.tiff', template_path + 'gplxcu_tmpname.tiff')
    gplxcu.addLineDesc({'idNumber':(0,0), 'dateOfBirth':(0,0), 'fullName':(0,0), 'Gender':(0,0), 'Dantoc':(0,0), 'NguyenQuan':(0,0)})    

    def __init__(self, supported_list=[cmnd9]):
        self.allcards = supported_list
        self.pool = Pool(processes=len(self.allcards)) 
    
    def abc(self, imgpath, rstext=sys.stdout):    
        img = cv2.imread(imgpath) #sys.argv[1] is image location
        if img is None:
            rstext.write('Input file is not an image.')
            sys.exit(0)
        rr= 600.0/img.shape[1]
        img = cv2.resize(img, None, fx=rr, fy=rr)
        blurness = calcBlur(img)
        if blurness < 50:
            rstext.write('Detected blurred image.')
            sys.exit(0)
        (h, w) = img.shape[:2]
        
        # Rotate image to deskew
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img00 = cv2.resize(img0[h/3:,w/3:],None,fx=0.5,fy=0.5) # scale down to estimate skew faster
        angle = estimate_skew_angle(img00,linspace(-5,5,41))
        print 'angle ', angle
        rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        img = cv2.warpAffine(img,rotM,(w,h))
    
        recognized_cards = []
        
        tt = time.time()
    #         for card in allcards:
    #             # Check if img is this card or not, if yes, how confident it is
    #             isCard, conf = card.findTextRegion(img)
    #             if isCard: recognized_cards.append((conf, card))
    #         print '1 ', time.time() -tt
        
        tt = time.time()
        allcards = [(c,img) for c in self.allcards]
        for (isCard, conf), card in self.pool.imap_unordered(fff, allcards):
            if isCard: recognized_cards.append((conf, card))
        print '2 ', time.time() -tt
        
        if len(recognized_cards) > 0:
            # Pick the most appropriate card
            _, foundcmnd = min(recognized_cards)
    #             foundcmnd.printResult(rstext)
            foundcmnd.extractLinesAndTexts(TensorFlowRecognizer(), out=rstext)
        else:
            rstext.write('No ID Card found.')            
               

if __name__ == '__main__':
    sys.argv = ['main.py','/home/loitg/Downloads/cmnd_data/6.jpg','/home/loitg/temp.txt']
    cmnd_path = '/home/loitg/Downloads/cmnd_data/realcapture/'
    p = CMNDPredictor()
    for fn in os.listdir(cmnd_path):
        if fn[-3:].upper() not in ['PEG','JPG','PNG']: continue
        p.abc(cmnd_path + fn, sys.stdout)
    
#     with codecs.open(sys.argv[2], 'w', encoding='utf-8') as rstext: #sys.argv[2] is output text
        
        


        
    