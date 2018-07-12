# -*- coding: utf-8 -*-
'''
Created on Oct 3, 2017

@author: loitg
'''

from utils.common import args, sharpen, sauvola, firstAnalyse
from utils.removedot import removedot
import cv2
from numpy.ctypeslib import ndpointer
from pylab import *
from scipy.ndimage import filters,interpolation,morphology,measurements, uniform_filter1d, maximum_filter1d, minimum_filter1d
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, minimum_filter
import ocrolib
from ocrolib import psegutils
from time import time

from ocrolib import morph,sl
from ocrolib.toplevel import *

from utils.common import DSHOW
from lineextract.line import Line
from algorithm import TensorFlowRecognizer.TensorFlowRecognizer


def pre_check_line(line):
    project = mean(1-line, axis=0)
    project = uniform_filter1d(project, line.shape[0]/3)
    m = mean(project)
    if (m > 0.13) & (1.0*line.shape[1]/line.shape[0] > 1.7):
        return True
    else:
        return False
    
def compute_boxmap(binary,scale,oriimg,threshold=(.5,4),dtype='i'):
    labels,n = morph.label(binary)
    objects = morph.find_objects(labels)
    boxmap = zeros(binary.shape,dtype)
    for i,o in enumerate(objects):
        h = sl.dim0(o)
        w = sl.dim1(o)
        ratio = float(h)/w if h > w else float(w)/h
        if h > 2*scale or h < scale/3:
            continue
        if ratio > 8: continue
#         if sl.area(o)**.5<threshold[0]*scale: continue
#         if sl.area(o)**.5>threshold[1]*scale: continue

        boxmap[o] = 1
    return boxmap

def compute_gradmaps(binary,scale):
    # use gradient filtering to find baselines
    binaryary = morph.r_opening(binary.astype(bool), (1,1))# CMND
    boxmap = compute_boxmap(binaryary,scale,binary)
    cleaned = boxmap*binaryary
    if args.usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(args.vscale*0.3*scale,
                                            args.hscale*6*scale),order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned,(max(4,args.vscale*0.3*scale),
                                            args.hscale*1.0*scale),order=(1,0))
        grad = uniform_filter(grad,(args.vscale,args.hscale*1*scale)) # CMND
    bottom = ocrolib.norm_max((grad<0)*(-grad))
#     bottom = minimum_filter(bottom,(2,6*scale))
    top = ocrolib.norm_max((grad>0)*grad)
#     top = minimum_filter(top,(2,6*scale))
    return bottom,top,boxmap

def compute_line_seeds(binaryary,bottom,top,scale):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = args.threshold
    vrange = int(args.vscale*scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked = bmarked*(bottom>t*amax(bottom)*t)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked = tmarked*(top>t*amax(top)*t/2)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = zeros(binaryary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
    DSHOW("lineseeds",[0.4*seeds,0.3*tmarked+0.7*bmarked,binaryary])
    return seeds

@checks(SEGMENTATION)
def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,sampling=[1,1], return_distances=1,return_indices=1) #CMND
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread

@checks(True,pad=int,expand=int,_=GRAYSCALE)
def extract_line(image,linedesc,pad=5):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    y0,x0,y1,x1 = [int(x) for x in [linedesc.bounds[0].start,linedesc.bounds[1].start, \
                  linedesc.bounds[0].stop,linedesc.bounds[1].stop]]
    y0,x0,y1,x1 = (y0-pad,x0-pad,y1+pad,x1+pad)
    h,w = image.shape[:2]
    y0 = clip(y0,0,h)
    y1 = clip(y1,0,h)
    x0 = clip(x0,0,w)
    x1 = clip(x1,0,w)
    if y0 < y1 and x0 < x1:
        if len(image.shape) == 2:
            return image[y0:y1, x0:x1]
        elif len(image.shape) == 3:
            return image[y0:y1, x0:x1, :]
    else:
        return None
    

class LinesExtractor(object):

    def __init__(self):
        pass
    
    def extractLines(self, imgpath):
        img_col = cv2.imread(imgpath)
        img_grey = cv2.imread(imgpath, 0)
        h,w = img_grey.shape
        img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.99, cv2.NORM_MINMAX)
        img_grey = sharpen(img_grey)
        img_bin_reversed = sauvola(img_grey, w=128, k=0.2, scaledown=args.binmaskscale, reverse=True)
#         ASHOW('ori', img_bin_reversed)
        objects, smalldot, scale = firstAnalyse(img_bin_reversed)
        dotremoved = removedot(img_bin_reversed, smalldot, scale)
        
        #these three coupled by self.originalpage and self.binpage, please DONOT break belows lines !!!
        self.originalpage = img_col
        self.binpage = dotremoved
        imglines = self._calc(objects, scale)  
        #########
         
        imgtexts = self.extractTexts(imglines, TensorFlowRecognizer())  
        
    
    
    
    def extractTexts(self, imglines, recognizer):
        texts = []
        for imgline in imglines:
            txt = recognizer.read(imgline.img)
            imgline.text = txt
            texts.append(txt)
        return texts
    
       
    def _calc(self, objects, scale):      
        if self.binpage is None:
            return
        tt=time()

        bottom,top,boxmap = compute_gradmaps(self.binpage,scale)
#         DSHOW('hihi', [0.5*bottom+0.5*top,self.binpage, boxmap])
        seeds0 = compute_line_seeds(self.binpage,bottom,top,scale)
        seeds,_ = morph.label(seeds0)

        llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
        spread = spread_labels(seeds,maxdist=scale)
        llabels = where(llabels>0,llabels,spread*self.binpage)
        segmentation = llabels*self.binpage     
        self.binpage = ocrolib.remove_noise(self.binpage,args.noise)
        bounds = psegutils.compute_lines(segmentation,scale)
        binpage_reversed = 1 - self.binpage
        lines = []
        for i,l in enumerate(bounds):
            imgline = extract_line(self.originalpage,l,pad=args.pad)
            if imgline is None: continue      
            line = Line(img=imgline, bounds=l)
            lines.append(line)
        return lines


        
        
        
if __name__ == '__main__':
    lm = LinesExtractor()
    
        
        
        
        
        
        
        
        
        
        
