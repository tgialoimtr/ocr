# -*- coding: utf-8 -*-
'''
Created on Oct 3, 2017

@author: loitg
'''

from pylab import *
import cv2
from scipy.ndimage import interpolation
from skimage.filters import threshold_sauvola, gaussian
from ocrolib import psegutils,morph,sl

cmnd_path = '/home/loitg/workspace/cmnd/scanned/'
cmnd_path = '/home/loitg/workspace/receipttest/img/'
hoadon_path = '/home/loitg/workspace/python/python/img/'
tmp_path = '/tmp/loitg/'

class obj:
    def __init__(self):
        pass
args = obj()
args.binmaskscale = 0.4
args.heavyprocesscale = 0.4
args.deskewscale = 0.1
args.range = 10

args.zoom = 0.5
args.range = 20
args.debug = 1
args.perc= 80
args.escale = 1.0
args.threshold = 0.5
args.lo = 5
args.hi = 90
args.usegauss = False
args.vscale = 1.0
args.hscale = 1.0
args.threshold = 0.2
args.pad = 0
args.expand = 3
args.model = '/home/loitg/workspace/receipttest/model/receipt-model-460-700-00590000.pyrnn.gz'
args.connect = 4
args.noise = 8
args.mode = 'cu'
args.template_path = '/home/loitg/Downloads/cmnd_data/template/'

def summarize(a):
    b = a.ravel()
    return a.dtype, a.shape, [amin(b),mean(b),amax(b)], percentile(b, [0,20,40,60,80,100])

def DSHOW(title,image):
    if not args.debug: return
    if type(image)==list:
        assert len(image)==3
        image = transpose(array(image),[1,2,0])
        if args.debug>0: imshow(image); ginput(timeout=-1)
    elif args.debug>0: 
        imshow(image, cmap='gray'); ginput(timeout=-1)
    
def ASHOW(title, image, scale=1.0, waitKey=False):
    HEIGHT = 600.0
    if len(image.shape) > 2:
        h,w,_ = image.shape
    else:
        h,w = image.shape
    canhlon = h if h > w else w
    tile = HEIGHT/canhlon
    
    mm = amax(image)
    if mm > 0:
        temp = image.astype(float)/mm
    else:
        temp = image.astype(float)
    
#     if len(image.shape) > 2:
#         temp = cv2.resize(temp,None,fx=tile*scale,fy=tile*scale)
#     else:
#         temp = interpolation.zoom(temp, tile*scale)
    temp = cv2.resize(temp,None,fx=tile*scale,fy=tile*scale)
    cv2.imshow(title, temp)
    if waitKey:
        cv2.waitKey(-1)

def sharpen(binimg):
#     blurred_l= gaussian(binimg,2)
    blurred_l= gaussian(binimg,0.8) #CMND
#     filter_blurred_l = gaussian(blurred_l, 1)
    filter_blurred_l = gaussian(blurred_l, 0.4)  # CMND
    alpha = 30
    return blurred_l + alpha * (blurred_l - filter_blurred_l) 
    
def estimate_skew_angle(image,angles):
    estimates = []
    binimage = sauvola(image, 11, 0.1).astype(float)
#     cv2.imshow('debug',binimage)
#     cv2.waitKey(-1)
    for a in angles:
        rotM = cv2.getRotationMatrix2D((binimage.shape[1]/2,binimage.shape[0]/2),a,1)
        rotated = cv2.warpAffine(binimage,rotM,(binimage.shape[1],binimage.shape[0]))
        v = mean(rotated,axis=1)
        d = [abs(v[i] - v[i-1]) for i in range(1,len(v))]
        d = var(d)
        estimates.append((d,a))
#     if args.debug>0:
#         plot([y for x,y in estimates],[x for x,y in estimates])
#         ginput(1,args.debug)
    _,a = max(estimates)
    return a

def sauvola(grayimg, w=51, k=0.2, scaledown=None, reverse=False):
    mask =None
    if scaledown is not None:
        mask = cv2.resize(grayimg,None,fx=scaledown,fy=scaledown)
        w = int(w * scaledown)
        if w % 2 == 0: w += 1
        mask = threshold_sauvola(mask, w, k)
        mask = cv2.resize(mask,(grayimg.shape[1],grayimg.shape[0]),fx=scaledown,fy=scaledown)
    else:
        if w % 2 == 0: w += 1
        mask = threshold_sauvola(grayimg, w, k)
    if reverse:
        return where(grayimg > mask, uint8(0), uint8(1))
    else:
        return where(grayimg > mask, uint8(1), uint8(0))
    
def simplefirstAnalyse(binary):
    binaryary = morph.r_closing(binary.astype(bool), (1,1))
    labels,_ = morph.label(binaryary)
    objects = morph.find_objects(labels) ### <<<==== objects here
    smalldot = zeros(binaryary.shape, dtype=binary.dtype)
    scale = int(binary.shape[0]*0.7)
    for i,o in enumerate(objects):       
        if (sl.width(o) < scale/2) or (sl.height(o) < scale/2):
            smalldot[o] = binary[o]
        if sl.dim0(o) > 3*scale:
            mask = where(labels[o] != (i+1),uint8(255),uint8(0))
            binary[o] = cv2.bitwise_and(binary[o],binary[o],mask=mask)
            continue
    return objects, smalldot, scale

def firstAnalyse(binary):
    binaryary = morph.r_closing(binary.astype(bool), (1,1))
    labels,_ = morph.label(binaryary)
    objects = morph.find_objects(labels) ### <<<==== objects here
    bysize = sorted(range(len(objects)), key=lambda k: sl.area(objects[k]))
#     bysize = sorted(objects,key=sl.area)
    scalemap = zeros(binaryary.shape)
    smalldot = zeros(binaryary.shape, dtype=binary.dtype)
    for i in bysize:
        o = objects[i]
        if amax(scalemap[o])>0: 
#             mask = where(labels[o] != (i+1),uint8(255),uint8(0))
#             binary[o] = cv2.bitwise_and(binary[o],binary[o],mask=mask)
            continue
        scalemap[o] = sl.area(o)**0.5
    scale = median(scalemap[(scalemap>3)&(scalemap<100)]) ### <<<==== scale here

    for i,o in enumerate(objects):       
        if (sl.width(o) < scale/2) or (sl.height(o) < scale/2):
            smalldot[o] = binary[o]
        if sl.dim0(o) > 3*scale:
            mask = where(labels[o] != (i+1),uint8(255),uint8(0))
            binary[o] = cv2.bitwise_and(binary[o],binary[o],mask=mask)
            continue
    return objects, smalldot, scale
        