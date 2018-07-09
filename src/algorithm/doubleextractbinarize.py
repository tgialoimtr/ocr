'''
Created on Jul 7, 2018

@author: loitg
'''
import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from utils.common import args, sharpen, sauvola, firstAnalyse
from utils.removedot import removedot

def pca2(x):
    X_std = x.astype(np.float64)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    return eig_vecs, eig_vals

def pca(x):
    X = x.astype(np.float64)
    num_data,dim = X.shape
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X
    print 'PCA - compact trick used'
    M = np.dot(X,X.T) #covariance matrix
    e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
    tmp = np.dot(X.T,EV).T #this is the compact trick
    V = tmp[::-1] #reverse since last eigenvectors are the ones we want
    S = np.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
    
    #return the projection matrix, the variance and the mean
    return V,S,mean_X

def gray2heatmap(img):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(img)
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

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
    
def abc(imgpath):
    print imgpath
    img_col = cv2.imread(imgpath)
    img_col = img_col.astype(np.float32)/255.0
    cv2.imshow('col', img_col)
    
    img_dn = img_col.reshape(-1,3)
    img_dn = img_dn
    V,S = pca2(img_dn)
#     print V
#     print S
#     
#     rs = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('bin', rs)
#     print 'bin', np.var(rs)
#     
#     u = stddirection(V[0,:])
#     rs = np.dot(img_col , u)
#     rs = cv2.normalize(rs, None, 0.0, 0.99, cv2.NORM_MINMAX)
#     cv2.imshow('bin0', rs)
#     print 'bin0', np.var(rs), u

    u = stddirection(V[:,0])
    rs = np.dot(img_col , u)
    rs = cv2.normalize(rs, None, 0.0, 0.99, cv2.NORM_MINMAX)
    cv2.imshow('bin1', rs)
    print 'bin1', np.var(rs), u
# 
#     u = stddirection(V[:,-1])
#     rs = np.dot(img_col , u)
#     rs = cv2.normalize(rs, None, 0.0, 0.99, cv2.NORM_MINMAX)
#     cv2.imshow('bin2', rs)
#     print 'bin2', np.var(rs), u
# 
#     u = stddirection(V[-1,:])
#     rs = np.dot(img_col , u)
#     rs = cv2.normalize(rs, None, 0.0, 0.99, cv2.NORM_MINMAX)
#     cv2.imshow('bin3', rs)
#     print 'bin3', np.var(rs), u

    img_grey = rs
    img_grey = sharpen(img_grey)
 
    img_bin_reversed = sauvola(img_grey, w=img_grey.shape[0]/2, k=0.05, reverse=True)
#         ASHOW('ori', img_bin_reversed)
    objects, smalldot, scale = firstAnalyse(img_bin_reversed)
    dotremoved = removedot(img_bin_reversed, smalldot, scale)
     
    cv2.imshow('bin', img_bin_reversed*255)
    print len(objects)
    cv2.imshow('dot', dotremoved*255)
    cv2.waitKey(-1)
    
if __name__ == '__main__':
    linedir = '/home/loitg/workspace/clocr/temp/0/'
    for linepath in os.listdir(linedir):
        abc(linedir+linepath)
    
