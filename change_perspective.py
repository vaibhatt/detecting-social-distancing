import cv2
import numpy as np


class change_perspective:
    def __init__(self,pts1,pts2,frame,shape):
        self.pts1 = pts1 #where pts1 and pts2 are lists comprising of four points
        self.pts2 = pts2
        self.frame = frame    
        self.shape = shape #shape in form of (width,height)

    def change(self):
        pts1 = np.float32(self.pts1)
        pts2 = np.float32(self.pts2)
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        result = cv2.warpPerspective(self.frame,matrix,self.shape)
        return result