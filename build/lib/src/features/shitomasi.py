import math 
from enum import Enum
import numpy as np 
import cv2




class ShiTomasiDetector(object): 
    def __init__(self, num_features=2000, quality_level = 0.01, min_coner_distance = 3):
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_coner_distance = min_coner_distance
        self.blockSize=5    # 3 is the default block size 

    def detect(self, frame, mask=None):                
        pts = cv2.goodFeaturesToTrack(frame, self.num_features, self.quality_level, self.min_coner_distance, blockSize=self.blockSize, mask=mask)
        # convert matrix of pts into list of keypoints 
        if pts is not None: 
            kps = [ cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts ]
        else:
            kps = []
        #if kVerbose:
        #    print('detector: Shi-Tomasi, #features: ', len(kps), ', #ref: ', self.num_features, ', frame res: ', frame.shape[0:2])      
        return kps
