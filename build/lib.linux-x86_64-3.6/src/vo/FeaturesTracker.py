import numpy as np
import cv2
from enum import Enum

from src.features.types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from src.features.matcher import FeatureMatcherTypes, BfFeatureMatcher, FlannFeatureMatcher
from src.features.manager import FeatureManager


class FeatureTrackerTypes(Enum):
    # Lucas Kanade pyramid optic flow (use pixel patch as "descriptor" and matching by optimization)
    LK = 0
    DES_BF = 1   # descriptor-based, brute force matching with knn
    DES_FLANN = 2   # descriptor-based, FLANN-based matching


class FeatureTrackingResult(object):
    def __init__(self):
        # all reference keypoints (numpy array Nx2)
        self.kps_ref = None
        # all current keypoints   (numpy array Nx2)
        self.kps_cur = None
        # all current descriptors (numpy array NxD)
        self.des_cur = None
        # indexes of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs_ref = None
        # indexes of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)
        self.idxs_cur = None
        # reference matched keypoints, kps_ref_matched = kps_ref[idxs_ref]
        self.kps_ref_matched = None
        # current matched keypoints, kps_cur_matched = kps_cur[idxs_cur]
        self.kps_cur_matched = None


class DescriptorFeatureTracker(object):
    def __init__(self, 
                 num_features=2000, 
                 num_levels=1, 
                 scale_factor=1.2, 
                 detector_type="FAST",
                 descriptor_type="ORB", 
                 match_ratio_test=0.7, 
                 tracker_type=FeatureTrackerTypes.DES_BF):

        self.feature_manager = FeatureManager(num_features=num_features,
                                              num_levels=num_levels,
                                              scale_factor=scale_factor,
                                              detector_type=detector_type,
                                              descriptor_type=descriptor_type)
        #
        
        self.tracker_type = tracker_type
        
        if self.tracker_type == FeatureTrackerTypes.DES_FLANN:

            self.matching_algo = FeatureMatcherTypes.FLANN

            self.matcher = FlannFeatureMatcher(norm_type=self.feature_manager.norm_type,
                                               cross_check=False,
                                               ratio_test=match_ratio_test,
                                               type=type)

        elif self.tracker_type == FeatureTrackerTypes.DES_BF:

            self.matching_algo = FeatureMatcherTypes.BF

            self.matcher = BfFeatureMatcher(norm_type=self.feature_manager.norm_type,
                                            cross_check=False,
                                            ratio_test=match_ratio_test,
                                            type=type)

        else:
            raise ValueError("Unmanaged feature tracker %s" %
                             self.tracker_type)

    # out: keypoints and descriptors

    def detectAndCompute(self, frame, mask=None):
        return self.feature_manager.detectAndCompute(frame, mask)

    # out: FeatureTrackingResult()

    def track(self, image_ref, image_cur, kps_ref, des_ref):
        kps_cur, des_cur = self.detectAndCompute(image_cur)
        
        # convert from list of keypoints to an array of points
        kps_cur = np.array([x.pt for x in kps_cur], dtype=np.float32)

        # knnMatch(queryDescriptors,trainDescriptors)
        idxs_ref, idxs_cur = self.matcher.match(des_ref, des_cur)

        res = FeatureTrackingResult()
        res.kps_ref = kps_ref  # all the reference keypoints
        res.kps_cur = kps_cur  # all the current keypoints
        res.des_cur = des_cur  # all the current descriptors

        res.kps_ref_matched = np.asarray(kps_ref[idxs_ref])  # the matched ref kps
        res.idxs_ref = np.asarray(idxs_ref)

        res.kps_cur_matched = np.asarray(kps_cur[idxs_cur])  # the matched cur kps
        res.idxs_cur = np.asarray(idxs_cur)

        return res
