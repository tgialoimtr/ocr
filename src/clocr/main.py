'''
Created on Jan 5, 2018

@author: loitg
'''
from utils.common import ASHOW

import numpy as np
import cv2, os, sys
from matplotlib import pyplot as plt
from time import time
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, minimum_filter

MIN_MATCH_COUNT = 10

img1 = cv2.imread('/home/loitg/Downloads/cmnd_data/template/template.tif',0)          # queryImage

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def xyz(img2):
    # ['clear', 'compute', 'defaultNorm', 'descriptorSize', 'descriptorType', 'detect', 
    #'detectAndCompute', 'empty', 'getDefaultName', 'getEdgeThreshold', 'getFastThreshold', 
    #'getFirstLevel', 'getMaxFeatures', 'getNLevels', 'getPatchSize', 'getScaleFactor', 
    #'getScoreType', 'getWTA_K', 'read', 'save', 'setEdgeThreshold', 'setFastThreshold', 
    #'setFirstLevel', 'setMaxFeatures', 'setNLevels', 'setPatchSize', 'setScaleFactor', 
    #'setScoreType', 'setWTA_K', 'write']
    orb = cv2.ORB_create(nfeatures=7000, scoreType=cv2.ORB_FAST_SCORE)
    tt = time()
    kp1, des1 = orb.detectAndCompute(img1,None)
#     print dir(kp1[0]) 'angle', 'class_id', 'octave', 'pt', 'response', 'size'
#     print des1[0].shape, des1[0].dtype
#     sys.exit(0)
    kp2, des2 = orb.detectAndCompute(img2,None)
    print 'compute ', time()-tt
    tt=time()    
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), \
            flags=cv2.DrawMatchesFlags_DEFAULT)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), \
            flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imshow('a', img1_kp)
    cv2.imshow('b', img2_kp)
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6,
                        key_size = 12, 
                        multi_probe_level = 1) 
    search_params = dict(checks = 50)
    print 'compute ', time()-tt
    tt=time()  
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    if len(matches) < 5: return 0
    # store all the good matches as per Lowe's ratio test.
    good = []
    for temp in matches:
        if len(temp) != 2: continue
        m,n = temp
        if m.distance < 0.7*n.distance:
            good.append(m)
    print 'good ',  len(good)
    print 'matching ', time()-tt
    tt=time()  
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        print 'inliers count ' + str(1.0*sum(mask)/mask.shape[0])
        print 'findHomography ', time()-tt
        tt=time()  
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
        img4 = four_point_transform(img2, dst.reshape(-1,2))
#         img4 = cv2.warpAffine(img2, Minverse, (w,h))   
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        return 0
#         matchesMask = None    

    draw_params = dict(matchColor = -1, # draw matches in green color
                       singlePointColor = (255,0,0),
    #                     matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good, None, **draw_params)
    img3 = cv2.resize(img3, None, fx=0.5,fy=0.5)
    cv2.imshow('ff', img3)
    cv2.imshow('gg', img4)
    cv2.waitKey(-1)
    
def abc(img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    tt=time()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print 'kp1 ', len(kp1)
    print 'kp2 ', len(kp2)
    print 'compute sift', time()-tt
    tt=time()
    cv2.imshow('img1', cv2.drawKeypoints(img1, kp1, None))
    cv2.imshow('icmg2', cv2.drawKeypoints(img2, kp2, None))
#     cv2.waitKey(-1)
    
    
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    print 'match ', time()-tt
    tt=time()
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    print 'good ',  len(good)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        print 'find transform matrix ', time()-tt
        tt=time()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    
    draw_params = dict(matchColor = -1, # draw matches in green color
                       singlePointColor = (255,0,0),
    #                     matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
    cv2.imshow('hhi', img3)
    cv2.waitKey(-1)
    
    
if __name__ == "__main__":
    cmndspath = '/home/loitg/Downloads/cmnd_data/realcapture/unique/'
    heightrange = range(img1.shape[0], img1.shape[0]*2, img1.shape[0]/6)
    for fn in os.listdir(cmndspath):
        if fn[-3:].upper() not in ['JPG','PEG','PNG']:
            continue
        p = cmndspath + fn
        for h in heightrange:
            print p
            img = cv2.imread(p,0)
            newwidth = img.shape[1] * h / img.shape[0]
            img = cv2.resize(img, (int(newwidth), h))
            xyz(img)
        



