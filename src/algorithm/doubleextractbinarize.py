'''
Created on Jul 7, 2018

@author: loitg
'''
import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from utils.common import args, sharpen, sauvola, firstAnalyse, estimate_skew_angle
from lineextract.ocropus.line_extractor import compute_line_seeds
from utils.common import gray2heatmap, pca2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter

def removedot(imgbin, smalldot, scale):   
#     ellipsis_map = zeros(imgbin.shape,dtype=uint8)
    horizental = maximum_filter(smalldot, (1,scale))
    horizental = minimum_filter(horizental, (1,scale))
    horizental = minimum_filter(horizental, (1,scale*3))
    horizental = maximum_filter(horizental, (1,scale*3))       
    linemap = horizental
    linemap = maximum_filter(linemap, (3,3))  
    ellipsis_map = cv2.bitwise_and(smalldot, smalldot, mask=linemap)
    return cv2.subtract(imgbin, ellipsis_map)

def binarize(img, black0, white0):
    img = img.astype(float)
    grey = np.ones_like(img, float)
    white = grey*white0
    black = grey*black0
    to_white = (img - white)**2
    to_white = np.sum(to_white, axis=2)
#     to_white = np.sqrt(to_white)
    to_black = (img - black)**2
    to_black = np.sum(to_black, axis=2)
#     to_black = np.sqrt(to_black)
    to_white = to_black + to_white
    to_black = to_black / to_white
    return to_black

def stddirection(a):
    if np.sum(a) < 0:
        return -a
    else:
        return a
    
def pipolarRotateExtractLine(img_col, expand):
    img_col = img_col.astype(np.float32)/255.0
#     cv2.imshow('col', img_col)
    
    img_dn = img_col.reshape(-1,3)
    img_dn = img_dn
    V,S = pca2(img_dn)
    u = stddirection(V[:,0])
    rs = np.dot(img_col , u)
    rs = cv2.normalize(rs, None, 0.0, 0.99, cv2.NORM_MINMAX)

    img_grey = rs
#     img_grey = sharpen(img_grey)
    scale = img_grey.shape[0]/4
    grad_grey = gaussian_filter(img_grey,(max(4,args.vscale*0.3*scale),
                                    args.hscale*5.0*scale),order=(1,0))
    grad_grey = cv2.normalize(abs(grad_grey), None, 0.0, 0.99, cv2.NORM_MINMAX)
    img_bin = sauvola(img_grey, w=img_grey.shape[0]/4, k=0.2, reverse=True)
    img_bin_reversed = img_bin.copy()
    
    def lambda_extract(o, arg_scale):
        h = o[0].stop - o[0].start
        w = o[1].stop - o[1].start
        return h < scale/2 and w < scale/2
    objects, extracts, scale = firstAnalyse(img_bin_reversed, [lambda_extract])
    dotremoved = cv2.subtract(img_bin_reversed, extracts[0])

    a, dotremoved = estimate_skew_angle(dotremoved,np.linspace(-2,2,21), binarize=False)
    
    ### OCROLIB
    grad_dot_bin = gaussian_filter(dotremoved.astype(np.float64),(max(4,args.vscale*0.3*scale),
                                args.hscale*5.0*scale),order=(1,0))
#     grad_dot_bin = cv2.normalize(abs(grad_dot_bin), None, 0.0, 0.99, cv2.NORM_MINMAX)
    
    bottom = cv2.normalize((grad_dot_bin<0)*(-grad_dot_bin), None, 0.0, 0.99, cv2.NORM_MINMAX)
    top = cv2.normalize((grad_dot_bin>0)*grad_dot_bin, None, 0.0, 0.99, cv2.NORM_MINMAX)
#     cv2.imshow('dot', dotremoved*240)

    ### PROJECTION PROFILE
    _,y0, y1 = extractLine(dotremoved, bottom, top)[0]
    rotM = cv2.getRotationMatrix2D((dotremoved.shape[1]/2,dotremoved.shape[0]/2),a,1)
    rotated_grey = cv2.warpAffine(img_grey,rotM,(dotremoved.shape[1],dotremoved.shape[0]))
    y0, y1 = expands(0, rotated_grey.shape[0], y0, y1, expand)
    line = rotated_grey[y0:y1, :]
    return line

def findPeak(array, threshold=0.5):
    peak_indices = []
    peak_values = []
    max_peak = 0
    for u in range (1,len(array)-1):
        if (array[u]>array[u-1]) and (array[u]>array[u+1]):
            val = array[u]
            if val > max_peak: max_peak = val
            peak_indices.append(u)
            peak_values.append(val)
    peak = zip(peak_values, peak_indices)
    top_peak_indices = [x[1] for x in peak if x[0] > max_peak*threshold ]
    return top_peak_indices
def expands(min_y, max_y, y0, y1, expand=0.1):
    d = y1 - y0
    n0 = y0 - int(d*expand); n1 = y1 + int(d*expand)
    if n0 < min_y: n0 = min_y
    if n1 > max_y: n1 = max_y
    return n0, n1
    
def projScore(top, bot, proj_ver):
#     n = proj_ver.shape[0]; d = bot - top
#     n0 = top - d/3; n1 = bot + d/3
#     if n0 < 0: n0 = 0
#     if n1 >= n: n = n-1
#     inner = np.mean(proj_ver[top:bot]) - np.std(proj_ver[top:bot])
#     outer = np.concatenate([proj_ver[n0:top], proj_ver[bot:n1]])
#     outer = np.mean(outer) + np.std(outer)
#     return 1.0 - 1.0*outer/inner
    return 0.5*(bot-top)/proj_ver.shape[0]

def extractLine(binimg, bottom, top):
    proj_ver = np.sum(binimg, axis=1)
    proj_bottom = np.sum(bottom, axis=1)
    proj_top = np.sum(top, axis=1)
    
    peak_bot = findPeak(proj_bottom) + [proj_ver.shape[0]-1]
    peak_top = findPeak(proj_top) + [0]
    peak_top.sort();peak_bot.sort();
    
    rs = []
    for top in peak_top:
        for bot in peak_bot:
            if top >= bot: continue
            proj_score = projScore(top, bot, proj_ver)
            score = proj_score + 0.5*proj_top[top]/255 + 0.5*proj_bottom[bot]/255
            rs.append((score, top, bot))
            break;
    rs.sort(reverse=True)
    return rs
if __name__ == '__main__':
    linedir = '/home/loitg/workspace/clocr/temp/0/'
    for linepath in os.listdir(linedir):
        line = pipolarRotateExtractLine(cv2.imread(linedir+linepath))
        cv2.imshow('line', line)
        cv2.waitKey(-1)
    
