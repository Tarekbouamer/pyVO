import sys 
import math 
from enum import Enum
import numpy as np 
import cv2

class RootSIFTFeature2D:
    def __init__(self, feature):
        # initialize the SIFT feature detector
        self.feature = feature

    def detect(self, frame, mask=None):
        return self.feature.detect(frame, mask)
 
    def transform_descriptors(self, des, eps=1e-7): 
        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)        
        return des 
            
    def compute(self, frame, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, des) = self.feature.compute(frame, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)

    # detect keypoints and their descriptors
    # out: kps, des 
    def detectAndCompute(self, frame, mask=None):
        # compute SIFT keypoints and descriptors
        (kps, des) = self.feature.detectAndCompute(frame, mask)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)