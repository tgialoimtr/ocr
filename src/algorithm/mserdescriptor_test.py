'''
Created on Aug 11, 2018

@author: loitg
'''
import cv2
import numpy as np
from time import time
from calc.keypoint_calc import four_point_transform
from cmnd.template import vocXml2Dict
from utils.common import TEMPORARY_PATH, RESOURCE_PATH
import mahotas
from scipy.spatial import distance as dist

# class StaticTemplate(object):
#     '''
#     classdocs
#     '''
# 
#     def __init__(self):
#         self.lines = {}
#         self.tpl_lines = {}
#         self.keypointCalc = None
#         self.img = None
#         self.desc = None
#         
#     
#     
#     @staticmethod
#     def createFrom(VOCxmlpath, fname, desc='Template', keypoint_type=KeyPointCalc.ORB):
#         _, vocrs =vocXml2Dict(VOCxmlpath)
#         tpl_lines = {}
#         if 'whole' not in vocrs: return None
#         abs_x0, abs_y0, abs_x1, abs_y1 = vocrs['whole']
#         template = StaticTemplate()
#         template.desc = desc
#         template.keypointCalc = KeyPointCalc(keypoint_type)
#         for tag, pos in vocrs.iteritems():
#             xmin, ymin, xmax, ymax = pos
#             tpl_lines[tag] = TemplateLine(xmin - abs_x0, ymin - abs_y0, xmax - xmin, ymax - ymin)    
#         img = cv2.imread(fname)
#         template.img = img[abs_y0:abs_y1, abs_x0:abs_x1]
#         template.tpl_lines = tpl_lines
#         template.kp, template.des = template.keypointCalc.calc(template.img) # here???
#         return template
#     
#     def extractLine(self, foundImg, linetmpl):
#         assert foundImg.shape[:2] == self.img.shape[:2]
#         x0 = linetmpl.relx
#         y0 = linetmpl.rely
#         x1 = linetmpl.relx + linetmpl.width
#         y1 = linetmpl.rely + linetmpl.height
#         x0 = np.clip(x0, 0, foundImg.shape[1])
#         x1 = np.clip(x1, 0, foundImg.shape[1])
#         y0 = np.clip(y0, 0, foundImg.shape[0])
#         y1 = np.clip(y1, 0, foundImg.shape[0])
#         bounds = (slice(y0, y1), slice(x0, x1))
#         line = foundImg[y0:y1, x0:x1, :]
#         line = Line(bounds, line, line.shape)
#         return line
#     
#     def find(self, img):
#         kp, des = self.keypointCalc.calc(img)
#         good = self.keypointCalc.match(self.des, des)
#         if len(good) > 10:
# #             draw_params = dict(matchColor = -1, # draw matches in green color
# #                                singlePointColor = (255,0,0),
# #             #                     matchesMask = matchesMask, # draw only inliers
# #                                flags = 2)
# #             img3 = cv2.drawMatches(self.img,self.kp,img,kp,good, None, **draw_params)
# #             cv2.imshow('gg', img3)
# #             cv2.waitKey(-1)
#             
#             lines = {}
#             src_pts = np.float32([ self.kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#             dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#             tt = time()
#             h,w = self.img.shape[:2]
#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#             
#             pts = np.float32([ [0,0],[w,0] ,[w,h],[0,h]]).reshape(-1,1,2)
#             dst = cv2.perspectiveTransform(pts,M)
#     
#             rs_img = four_point_transform(img, dst.reshape(-1,2))
#             rs_img = cv2.resize(rs_img, (w,h))
# #             print 'findHomography ', time()-tt
# #             print 'mean', np.mean(mask)
# #             cv2.imshow('img', rs_img)
# #             cv2.waitKey(-1)
#             for tag, linetempl in self.tpl_lines.iteritems():
#                 line = self.extractLine(rs_img, linetempl)
#                 lines[tag] = line
#             return np.mean(mask), lines
#         else:
#             return 0, None  


 
class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius
 
    def describe(self, image):
        image = cv2.resize(image, (2*self.radius, 2*self.radius))
        return mahotas.features.zernike_moments(image, self.radius)
         
class SceneTextKeyPointCalc(object):
    def __init__(self):
        self.mser = cv2.MSER_create()
        self.erc1 = cv2.text.loadClassifierNM1(RESOURCE_PATH + 'trained_classifierNM1.xml')
        self.zernikeDesc = ZernikeMoments(5)
    
    def calc(self, img):
        channels = cv2.text.computeNMChannels(img)
    #     cv2.Laplacian(temp, cv2.CV_64F).var()
        vis = img.copy()
        i=0
        for i,channel in enumerate(channels):
            er1 = cv2.text.createERFilterNM1(self.erc1,60,0.000015,0.0004,0.5,True,0.7)
        #     er2 = cv2.text.createERFilterNM2(erc2,0.5)
            
            regions = cv2.text.detectRegions(channel,er1,None)
            
            
            
            
            for points in regions:
                i +=1
#                 cv2.fillConvexPoly(vis, points, (255*(i%3), 255*((i+1)%3), 255*((i+2)%3)))
                (x,y),radius = cv2.minEnclosingCircle(points)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(vis, center,radius,(0,255,0),2)
                outline = np.zeros((2*radius+1, 2*radius+1), dtype = "uint8")
                points = points - np.array([center[0]-radius, center[1]-radius])
                cv2.drawContours(outline, [points], -1, 255, -1)
                cv2.imshow('shape', outline)
                k = cv2.waitKey(-1)
                print chr(k), ' with height ',  str(2*radius+1)
#                 print cv2.HuMoments(cv2.moments(outline)).flatten()
                zernikeMoment = self.zernikeDesc.describe(outline)
                print zernikeMoment
        
        kp=None
        des=None
        print 'total ', str(i)
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
#         print 'good ',  len(good)
#         print 'matching ', time()-tt
        return good
    


if __name__ == '__main__':
    kpcalc = SceneTextKeyPointCalc()
    img = cv2.imread(TEMPORARY_PATH + 'template/image/002_L0360430010_0.jpg')
    kpcalc.calc(img)
    
    