'''
Created on Jan 5, 2018

@author: loitg
'''
from utils.common import ASHOW

import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import time
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, minimum_filter

MIN_MATCH_COUNT = 10

img1 = cv2.imread('/home/loitg/Downloads/hihi/AJISENJAPANESERESTAURANT_5.jpg',0)          # queryImage
img1 = minimum_filter(img1,(10,10))
img1 = cv2.blur(img1, (5,5))
img2 = cv2.imread('/home/loitg/Downloads/hihi/AJISENJAPANESERESTAURANT_9.jpg',0) # trainImage
img2 = minimum_filter(img2,(10,10))
img2 = cv2.blur(img2, (5,5))

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
tt=time()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print 'kp1 ', len(kp1)
print 'kp2 ', len(kp2)
print time()-tt
cv2.imshow('img1', cv2.drawKeypoints(img1, kp1, None))
cv2.imshow('icmg2', cv2.drawKeypoints(img2, kp2, None))
cv2.waitKey(-1)



FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

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

plt.imshow(img3, 'gray'),plt.show()


