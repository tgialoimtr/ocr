'''
Created on May 21, 2018

@author: loitg
'''
import os
import cv2
import numpy as np
from bs4 import BeautifulSoup
from calc.template_calc import *
from calc.keypoint_calc import KeyPointCalc, four_point_transform
from time import time

from lineextract.line import Line

class TemplateLine(object):
    def __init__(self, relx, rely, width, height):
        self.width = width
        self.height = height
        self.relx = relx
        self.rely = rely

def vocXml2Dict(VOCxmlpath):
    vocrs = {}
    with open(VOCxmlpath) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    anno = BeautifulSoup(xml, 'lxml')
    objs = anno.findAll('object')
    fname = anno.findChild('path').contents[0]
    for obj in objs:
        obj_names = obj.findChildren('name')
        name_tag = obj_names[0]
        tag = str(name_tag.contents[0])
        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        vocrs[tag] = (xmin, ymin, xmax, ymax)
        
    return fname, vocrs
             
class StaticTemplate(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.lines = {}
        self.tpl_lines = {}
        self.keypointCalc = None
        self.img = None
        
    
    
    @staticmethod
    def createFrom(VOCxmlpath, keypoint_type=KeyPointCalc.ORB):
        fname, vocrs =vocXml2Dict(VOCxmlpath)
        print fname
        print vocrs
        tpl_lines = {}
        if 'whole' not in vocrs: return None
        abs_x0, abs_y0, abs_x1, abs_y1 = vocrs['whole']
        template = StaticTemplate()
        template.keypointCalc = KeyPointCalc(keypoint_type)
        for tag, pos in vocrs.iteritems():
            xmin, ymin, xmax, ymax = pos
            tpl_lines[tag] = TemplateLine(xmin - abs_x0, ymin - abs_y0, xmax - xmin, ymax - ymin)    
        img = cv2.imread(fname)
        template.img = img[abs_y0:abs_y1, abs_x0:abs_x1]
        template.tpl_lines = tpl_lines
        template.kp, template.des = template.keypointCalc.calc(template.img) # here???
        return template
    
    def extractLine(self, foundImg, linetmpl):
        assert foundImg.shape[:2] == self.img.shape[:2]
        x0 = linetmpl.relx
        y0 = linetmpl.rely
        x1 = linetmpl.relx + linetmpl.width
        y1 = linetmpl.rely + linetmpl.height
        x0 = np.clip(x0, 0, foundImg.shape[1])
        x1 = np.clip(x1, 0, foundImg.shape[1])
        y0 = np.clip(y0, 0, foundImg.shape[0])
        y1 = np.clip(y1, 0, foundImg.shape[0])
        bounds = (slice(y0, y1), slice(x0, x1))
        line = foundImg[y0:y1, x0:x1, :]
        line = Line(bounds, line, line.shape)
        return line
    
    def find(self, img):
        kp, des = self.keypointCalc.calc(img)
        good = self.keypointCalc.match(self.des, des)
        if len(good) > 10:
#             draw_params = dict(matchColor = -1, # draw matches in green color
#                                singlePointColor = (255,0,0),
#             #                     matchesMask = matchesMask, # draw only inliers
#                                flags = 2)
#             img3 = cv2.drawMatches(self.img,self.kp,img,kp,good, None, **draw_params)
#             cv2.imshow('gg', img3)
#             cv2.waitKey(-1)
            
            lines = {}
            src_pts = np.float32([ self.kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            tt = time()
            h,w = self.img.shape[:2]
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            
            pts = np.float32([ [0,0],[w,0] ,[w,h],[0,h]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
    
            rs_img = four_point_transform(img, dst.reshape(-1,2))
            rs_img = cv2.resize(rs_img, (w,h))
            print 'findHomography ', time()-tt
            print 'mean', np.mean(mask)
#             cv2.imshow('img', rs_img)
#             cv2.waitKey(-1)
            for tag, linetempl in self.tpl_lines.iteritems():
                line = self.extractLine(rs_img, linetempl)
                lines[tag] = line
            return np.mean(mask), lines
        else:
            return 0, None       
    
    
if __name__ == '__main__':
    cancuoc_tmpl = StaticTemplate.createFrom('/home/loitg/Downloads/cmnd_data/template/0cancuoc.xml')
    samplespath = '/home/loitg/Downloads/cmnd_data/moi/'
    for fn in os.listdir(samplespath):
#         fn = '07.jpg'
        if fn[-3:].upper() not in ['PEG', 'JPG', 'PNG']: continue
        print fn + '-------------------'
        sample = cv2.imread(samplespath + fn)
#         sample = cv2.GaussianBlur(sample, (5,5), 0)
#         M= cv2.getRotationMatrix2D((sample.shape[1]/2, sample.shape[0]/2), 135, 1)
#         sample = cv2.warpAffine(sample, M ,(sample.shape[1], sample.shape[0]))
        prob, lines = cancuoc_tmpl.find(sample)
        if lines is None: continue
#         for tag, line in lines.iteritems():
#             print tag + str(line.img.shape)
#             cv2.imshow('line', line.img)
#             cv2.waitKey(-1)
    
    