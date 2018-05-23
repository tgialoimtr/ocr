'''
Created on May 21, 2018

@author: loitg
'''
import cv2
import numpy as np
from time import time

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


class KeyPointCalc(object):
    '''
    classdocs
    '''
    ORB = 'ORB'

    def __init__(self, kptype=ORB):
        '''
        Constructor
        '''
        if kptype == self.ORB:
            self.builder = cv2.ORB_create(nfeatures=7000, scoreType=cv2.ORB_FAST_SCORE)
        elif kptype == self.SIFT:
            self.builder = cv2.xfeatures2d.SIFT_create()
    
    def calc(self, img):
        kp, des = self.builder.detectAndCompute(img, None)
        return kp, des
    
    def match(self, des1, des2):
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6,
                            key_size = 12, 
                            multi_probe_level = 1) 
        search_params = dict(checks = 50)
        tt=time()  
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        if len(matches) < 5: return []
        # store all the good matches as per Lowe's ratio test.
        good = []
        for temp in matches:
            if len(temp) != 2: continue
            m,n = temp
            if m.distance < 0.7*n.distance:
                good.append(m)
        print 'good ',  len(good)
        print 'matching ', time()-tt
        return good