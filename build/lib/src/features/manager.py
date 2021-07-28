import sys
import math
from enum import Enum
import numpy as np
import cv2
from collections import Counter

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from src.features.types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo

from src.utils.sys import Printer, import_from
from src.utils.features import unpackSiftOctaveKps, UnpackOctaveMethod, sat_num_features, kdt_nms, ssc_nms, octree_nms, grid_nms
from src.utils.geometric import hamming_distance, hamming_distances, l2_distance, l2_distances

from src.features.root_sift import RootSIFTFeature2D
from src.features.shitomasi import ShiTomasiDetector


kVerbose = True
kNumFeatureDefault = 2000
kNumLevelsDefault = 4
kScaleFactorDefault = 1.2
kNumLevelsInitSigma = 40
kSigmaLevel0 = 1.0
kDrawOriginalExtractedFeatures = False  # for debugging
# 7 is the standard keypoint size on layer 0  => actual size = 7*kFASTKeyPointSizeRescaleFactor
kFASTKeyPointSizeRescaleFactor = 4
# 7 is the standard keypoint size on layer 0  => actual size = 7*kAGASTKeyPointSizeRescaleFactor
kAGASTKeyPointSizeRescaleFactor = 4
# 5 is the selected keypoint size on layer 0 (see below) => actual size = 5*kShiTomasiKeyPointSizeRescaleFactor
kShiTomasiKeyPointSizeRescaleFactor = 5



class KeyPointFilterTypes(Enum):
    NONE         = 0
    SAT          = 1      # sat the number of features (keep the best N features: 'best' on the basis of the keypoint.response)
    KDT_NMS      = 2      # Non-Maxima Suppression based on kd-tree
    SSC_NMS      = 3      # Non-Maxima Suppression based on https://github.com/BAILOOL/ANMS-Codes
    OCTREE_NMS   = 4      # Distribute keypoints by using a octree (as a matter of fact, a quadtree): from ORBSLAM2
    GRID_NMS     = 5      # NMS by using a grid 
    
    
class FeatureManager(object):
    def __init__(self, 
                 num_features=2000,
                 num_levels=1,
                 scale_factor=1.2,
                 detector_type=FeatureDetectorTypes.FAST,
                 descriptor_type=FeatureDescriptorTypes.ORB):

        self.detector_type = detector_type
        self._feature_detector = None

        self.descriptor_type = descriptor_type
        self._feature_descriptor = None

        # main feature manager properties
        self.num_features = num_features
        self.num_levels = num_levels
        # not always applicable = > 0: start pyramid from input image;
        self.first_level = 0
        #                          -1: start pyramid from up-scaled image*scale_factor (as in SIFT)
        self.scale_factor = scale_factor  # scale factor bewteen two octaves
        self.sigma_level0 = kSigmaLevel0  # sigma on first octave
        
        # for methods that uses octaves (SIFT, SURF, etc)
        self.layers_per_octave = 3

        # feature norm options
        self.norm_type = None            # descriptor norm type
        
        # pointer to a function for computing the distance between two points
        self.descriptor_distance = None
        
        # pointer to a function for computing the distances between two array of corresponding points
        self.descriptor_distances = None

        # block adaptor options
        self.use_bock_adaptor = False
        self.block_adaptor = None

        # pyramid adaptor options: at present time pyramid adaptor has the priority and can combine a block adaptor withint itself
        self.use_pyramid_adaptor = False
        self.pyramid_adaptor = None
        self.pyramid_type = PyramidType.RESIZE
        self.pyramid_do_parallel = True
        # if pyramid adaptor is active, one can require to compute a certain number of features per level (see PyramidAdaptor)
        self.do_sat_features_per_level = False
        # automatically managed below depending on features
        self.force_multiscale_detect_and_compute = False

        # automatically managed below depending on selected features
        self.oriented_features = True
        # automatically managed below depending on selected features
        self.do_keypoints_size_rescaling = False
        # automatically managed below depending on selected features
        self.need_color_image = False

        # default keypoint-filter type
        self.keypoint_filter_type = KeyPointFilterTypes.SAT
        
        # need or not non-maximum suppression of keypoints
        self.need_nms = False
        
        # default keypoint-filter type if NMS is needed
        self.keypoint_nms_filter_type = KeyPointFilterTypes.KDT_NMS

        # initialize sigmas for keypoint levels (used for SLAM)
        self.init_sigma_levels()

        # --------------------------------------------- #
        # manage different opencv versions
        # --------------------------------------------- #
        print("using opencv ", cv2.__version__)
        
        # check opencv version in order to use the right modules
        opencv_major = int(cv2.__version__.split('.')[0])
        opencv_minor = int(cv2.__version__.split('.')[1])
        if opencv_major == 3:
            SIFT_create     = import_from('cv2', 'SIFT_create')
            SURF_create     = import_from('cv2.xfeatures2d', 'SURF_create')
            FREAK_create    = import_from('cv2.xfeatures2d', 'FREAK_create')
            ORB_create      = import_from('cv2', 'ORB_create')
            BRISK_create    = import_from('cv2', 'BRISK_create')
            KAZE_create     = import_from('cv2', 'KAZE_create')
            AKAZE_create    = import_from('cv2', 'AKAZE_create')
            BoostDesc_create = import_from('cv2', 'xfeatures2d_BoostDesc', 'create')
            # found but it does not work! (it does not find the .create() method)
            MSD_create      = import_from('cv2', 'xfeatures2d_MSDDetector')
            # Affine_create = import_from('cv2','xfeatures2d_AffineFeature2D') # not found
            DAISY_create    = import_from('cv2', 'xfeatures2d_DAISY', 'create')
            STAR_create     = import_from('cv2', 'xfeatures2d_StarDetector', 'create')
            HL_create       = import_from('cv2', 'xfeatures2d_HarrisLaplaceFeatureDetector', 'create')
            LATCH_create    = import_from('cv2', 'xfeatures2d_LATCH', 'create')
            LUCID_create    = import_from('cv2', 'xfeatures2d_LUCID', 'create')
            VGG_create       = import_from('cv2', 'xfeatures2d_VGG', 'create')
            BEBLID_create   = import_from('cv2', 'xfeatures2d', 'BEBLID_create')
             
        elif opencv_major == 4 and opencv_minor >= 5:
            SIFT_create     = import_from('cv2', 'SIFT_create')
            SURF_create     = import_from('cv2.xfeatures2d', 'SURF_create')
            FREAK_create    = import_from('cv2.xfeatures2d', 'FREAK_create')
            ORB_create      = import_from('cv2', 'ORB_create')
            BRISK_create = import_from('cv2', 'BRISK_create')
            KAZE_create = import_from('cv2', 'KAZE_create')
            AKAZE_create = import_from('cv2', 'AKAZE_create')
            BoostDesc_create = import_from('cv2', '', 'create')
            MSD_create = import_from('cv2', 'xfeatures2d_MSDDetector')
            DAISY_create = import_from('cv2', 'xfeatures2d_DAISY', 'create')
            STAR_create = import_from('cv2', 'xfeatures2d_StarDetector', 'create')
            HL_create = import_from('cv2', 'xfeatures2d_HarrisLaplaceFeatureDetector', 'create')
            LATCH_create = import_from('cv2', 'xfeatures2d_LATCH', 'create')
            LUCID_create = import_from('cv2', 'xfeatures2d_LUCID', 'create')
            VGG_create = import_from('cv2', 'xfeatures2d_VGG', 'create')
            BEBLID_create = import_from('cv2', 'xfeatures2d', 'BEBLID_create')
        
        else:
            SIFT_create = import_from('cv2.xfeatures2d', 'SIFT_create')
            SURF_create = import_from('cv2.xfeatures2d', 'SURF_create')
            FREAK_create = import_from('cv2.xfeatures2d', 'FREAK_create')
            ORB_create = import_from('cv2', 'ORB')
            BRISK_create = import_from('cv2', 'BRISK')
            KAZE_create = import_from('cv2', 'KAZE')
            AKAZE_create = import_from('cv2', 'AKAZE')
            BoostDesc_create = import_from('cv2', 'xfeatures2d_BoostDesc', 'create')
            MSD_create = import_from('cv2', 'xfeatures2d_MSDDetector')
            DAISY_create = import_from('cv2', 'xfeatures2d_DAISY', 'create')
            STAR_create = import_from('cv2', 'xfeatures2d_StarDetector', 'create')
            HL_create = import_from('cv2', 'xfeatures2d_HarrisLaplaceFeatureDetector', 'create')
            LATCH_create = import_from('cv2', 'xfeatures2d_LATCH', 'create')
            LUCID_create = import_from('cv2', 'xfeatures2d_LUCID', 'create')
            VGG_create = import_from('cv2', 'xfeatures2d_VGG', 'create')
            BEBLID_create = import_from('cv2', 'xfeatures2d', 'BEBLID_create')

        # pure detectors
        self.FAST_create    = import_from('cv2', 'FastFeatureDetector_create')
        self.AGAST_create   = import_from('cv2', 'AgastFeatureDetector_create')
        self.GFTT_create    = import_from('cv2', 'GFTTDetector_create')
        self.MSER_create    = import_from('cv2', 'MSER_create')
        self.MSD_create     = MSD_create
        self.STAR_create    = STAR_create
        self.HL_create      = HL_create
        
        # detectors and descriptors
        self.SIFT_create    = SIFT_create
        self.SURF_create    = SURF_create
        self.ORB_create     = ORB_create
        self.BRISK_create   = BRISK_create
        self.AKAZE_create   = AKAZE_create
        self.KAZE_create    = KAZE_create
        
        # pure descriptors
        self.FREAK_create       = FREAK_create      #  only descriptor
        self.BoostDesc_create   = BoostDesc_create
        self.DAISY_create       = DAISY_create
        self.LATCH_create       = LATCH_create
        self.LUCID_create       = LUCID_create
        self.VGG_create         = VGG_create
        self.BEBLID_create      = BEBLID_create

        # --------------------------------------------- #
        # check if we want descriptor == detector
        # --------------------------------------------- #
        self.is_detector_equal_to_descriptor = (self.detector_type.name == self.descriptor_type.name)

        # N.B.: the following descriptors assume keypoint.octave extacly represents an octave with a scale_factor=2
        #       and not a generic level with scale_factor < 2
        if self.descriptor_type in [
            # [NOK] SIFT seems to assume the use of octaves (https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp#L1128)
            FeatureDescriptorTypes.SIFT,
            FeatureDescriptorTypes.ROOT_SIFT,  # [NOK] same as SIFT
            # FeatureDescriptorTypes.SURF,      # [OK]  SURF computes the descriptor by considering the keypoint.size (https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/surf.cpp#L600)
            # [NOK] AKAZE does NOT seem to compute the right scale index for each keypoint.size (https://github.com/opencv/opencv/blob/master/modules/features2d/src/kaze/AKAZEFeatures.cpp#L1508)
            FeatureDescriptorTypes.AKAZE,
            # [NOK] similar to KAZE
            FeatureDescriptorTypes.KAZE,
            # FeatureDescriptorTypes.FREAK,     # [OK]  FREAK computes the right scale index for each keypoint.size (https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/freak.cpp#L468)
            # FeatureDescriptorTypes.BRISK      # [OK]  BRISK computes the right scale index for each keypoint.size (https://github.com/opencv/opencv/blob/master/modules/features2d/src/brisk.cpp#L697)
            # FeatureDescriptorTypes.BOOST_DESC # [OK]  BOOST_DESC seems to properly rectify each keypoint patch size (https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/boostdesc.cpp#L346)
        ]:
            # the above descriptors work on octave layers with a scale_factor=2!
            self.scale_factor = 2
            Printer.orange('forcing scale factor=2 for detector',
                           self.descriptor_type.name)

        self.orb_params = dict(nfeatures=num_features,
                               scaleFactor=self.scale_factor,
                               nlevels=self.num_levels,
                               patchSize=31,
                               edgeThreshold=10,  # 31, #19, #10,   # margin from the frame border
                               fastThreshold=20,
                               firstLevel=self.first_level,
                               WTA_K=2,
                               scoreType=cv2.ORB_FAST_SCORE)  # scoreType=cv2.ORB_HARRIS_SCORE, scoreType=cv2.ORB_FAST_SCORE

        # --------------------------------------------- #
        # init detector
        # --------------------------------------------- #
        if self.detector_type == FeatureDetectorTypes.SIFT or self.detector_type == FeatureDetectorTypes.ROOT_SIFT:
            sift = self.SIFT_create(nOctaveLayers=self.layers_per_octave)
            self.set_sift_parameters()
            
            if self.detector_type == FeatureDetectorTypes.ROOT_SIFT:
                self._feature_detector = RootSIFTFeature2D(sift)
            
            else:
                self._feature_detector = sift
            #
            #
        elif self.detector_type == FeatureDetectorTypes.SURF:
            self._feature_detector = self.SURF_create(
                nOctaves=self.num_levels, nOctaveLayers=self.layers_per_octave)
            # self.intra_layer_factor = 1.2599   # num layers = nOctaves*nOctaveLayers  scale=2^(1/nOctaveLayers) = 1.2599
            self.scale_factor = 2               # force scale factor = 2 between octaves
            #
            #
        elif self.detector_type == FeatureDetectorTypes.ORB:
            self._feature_detector = self.ORB_create(**self.orb_params)
            self.use_bock_adaptor = True          # add a block adaptor?
            # ORB tends to generate overlapping keypoint on different levels <= KDT NMS seems to be very useful here!
            self.need_nms = self.num_levels > 1
            #
            #
        elif self.detector_type == FeatureDetectorTypes.ORB2:
            orb2_num_levels = self.num_levels
            self._feature_detector = Orbslam2Feature2D(
                self.num_features, self.scale_factor, orb2_num_levels)
            # ORB2 cpp implementation already includes the algorithm OCTREE_NMS
            self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.BRISK:
            self._feature_detector = self.BRISK_create(octaves=self.num_levels)
            # self.intra_layer_factor = 1.3          # from the BRISK opencv code this seems to be the used scale factor between intra-octave frames
            # self.intra_layer_factor = math.sqrt(2) # approx, num layers = nOctaves*nOctaveLayers, from the BRISK paper there are octave ci and intra-octave di layers, t(ci)=2^i, t(di)=2^i * 1.5
            self.scale_factor = 2                   # force scale factor = 2 between octaves
            #self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.KAZE:
            self._feature_detector = self.KAZE_create(
                nOctaves=self.num_levels, threshold=0.0005)  # default: threshold = 0.001f
            self.scale_factor = 2                  # force scale factor = 2 between octaves
            #
            #
        elif self.detector_type == FeatureDetectorTypes.AKAZE:
            self._feature_detector = self.AKAZE_create(
                nOctaves=self.num_levels, threshold=0.0005)  # default: threshold = 0.001f
            self.scale_factor = 2                  # force scale factor = 2 between octaves
            #
            #
        elif self.detector_type == FeatureDetectorTypes.SUPERPOINT:
            self.oriented_features = False
            self._feature_detector = SuperPointFeature2D()
            if self.descriptor_type != FeatureDescriptorTypes.NONE:
                self.use_pyramid_adaptor = self.num_levels > 1
                self.need_nms = self.num_levels > 1
                self.pyramid_type = PyramidType.GAUSS_PYRAMID
                # N.B.: SUPERPOINT interface class is not thread-safe!
                self.pyramid_do_parallel = False
                # force it since SUPERPOINT cannot compute descriptors separately from keypoints
                self.force_multiscale_detect_and_compute = True
            #
            #
        elif self.detector_type == FeatureDetectorTypes.FAST:
            self.oriented_features = False
            self._feature_detector = self.FAST_create(
                threshold=20, nonmaxSuppression=True)
            if self.descriptor_type != FeatureDescriptorTypes.NONE:
                # self.use_bock_adaptor = True  # override a block adaptor?
                self.use_pyramid_adaptor = self.num_levels > 1   # override a pyramid adaptor?
                #self.pyramid_type = PyramidType.GAUSS_PYRAMID
                #self.first_level = 0
                #self.do_sat_features_per_level = True
                self.need_nms = self.num_levels > 1
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
                self.do_keypoints_size_rescaling = True
            #
            #
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI:
            self.oriented_features = False
            self._feature_detector = ShiTomasiDetector(self.num_features)
            if self.descriptor_type != FeatureDescriptorTypes.NONE:
                # self.use_bock_adaptor = False  # override a block adaptor?
                self.use_pyramid_adaptor = self.num_levels > 1
                #self.pyramid_type = PyramidType.GAUSS_PYRAMID
                #self.first_level = 0
                self.need_nms = self.num_levels > 1
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
                self.do_keypoints_size_rescaling = True
            #
            #
        elif self.detector_type == FeatureDetectorTypes.AGAST:
            self.oriented_features = False
            self._feature_detector = self.AGAST_create(
                threshold=10, nonmaxSuppression=True)
            if self.descriptor_type != FeatureDescriptorTypes.NONE:
                # self.use_bock_adaptor = True  # override a block adaptor?
                self.use_pyramid_adaptor = self.num_levels > 1   # override a pyramid adaptor?
                #self.pyramid_type = PyramidType.GAUSS_PYRAMID
                #self.first_level = 0
                self.need_nms = self.num_levels > 1
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
                self.do_keypoints_size_rescaling = True
            #
            #
        elif self.detector_type == FeatureDetectorTypes.GFTT:
            self.oriented_features = False
            self._feature_detector = self.GFTT_create(
                self.num_features, qualityLevel=0.01, minDistance=3, blockSize=5, useHarrisDetector=False, k=0.04)
            if self.descriptor_type != FeatureDescriptorTypes.NONE:
                # self.use_bock_adaptor = True  # override a block adaptor?
                self.use_pyramid_adaptor = self.num_levels > 1   # override a pyramid adaptor?
                #self.pyramid_type = PyramidType.GAUSS_PYRAMID
                #self.first_level = 0
                self.need_nms = self.num_levels > 1
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
                self.do_keypoints_size_rescaling = True
            #
            #
        elif self.detector_type == FeatureDetectorTypes.MSER:
            self._feature_detector = self.MSER_create()
            # self.use_bock_adaptor = True  # override a block adaptor?
            self.use_pyramid_adaptor = self.num_levels > 1   # override a pyramid adaptor?
            # parallel computations generate segmentation fault (is MSER thread-safe?)
            self.pyramid_do_parallel = False
            #self.pyramid_type = PyramidType.GAUSS_PYRAMID
            #self.first_level = 0
            self.need_nms = self.num_levels > 1
            #self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
            #
            #
        elif self.detector_type == FeatureDetectorTypes.MSD:
            #detector = ShiTomasiDetector(self.num_features)
            #self._feature_detector = self.MSD_create(detector)
            self._feature_detector = self.MSD_create()
            print('MSD detector info:', dir(self._feature_detector))
            # self.use_bock_adaptor = True  # override a block adaptor?
            # self.use_pyramid_adaptor = self.num_levels > 1   # override a pyramid adaptor?
            #self.pyramid_type = PyramidType.GAUSS_PYRAMID
            #self.first_level = 0
            #self.need_nms = self.num_levels > 1
            #self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
            #
            #
        elif self.detector_type == FeatureDetectorTypes.STAR:
            self.oriented_features = False
            self._feature_detector = self.STAR_create(maxSize=45,
                                                      responseThreshold=10,  # =30
                                                      lineThresholdProjected=10,
                                                      lineThresholdBinarized=8,
                                                      suppressNonmaxSize=5)
            if self.descriptor_type != FeatureDescriptorTypes.NONE:
                # self.use_bock_adaptor = True  # override a block adaptor?
                self.use_pyramid_adaptor = self.num_levels > 1   # override a pyramid adaptor?
                #self.pyramid_type = PyramidType.GAUSS_PYRAMID
                #self.first_level = 0
                #self.need_nms = self.num_levels > 1
                #self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
            #
            #
        elif self.detector_type == FeatureDetectorTypes.HL:
            self.oriented_features = False
            self._feature_detector = self.HL_create(numOctaves=self.num_levels,
                                                    corn_thresh=0.005,  # = 0.01
                                                    DOG_thresh=0.01,  # = 0.01
                                                    maxCorners=self.num_features,
                                                    num_layers=4)  #
            self.scale_factor = 2   # force scale factor = 2 between octaves
            #
            #
        elif self.detector_type == FeatureDetectorTypes.D2NET:
            self.need_color_image = True
            self.num_levels = 1  # force unless you have 12GB of VRAM
            multiscale = self.num_levels > 1
            self._feature_detector = D2NetFeature2D(multiscale=multiscale)
            #self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.DELF:
            self.need_color_image = True
            # self.num_levels = 1 # force              #scales are computed internally
            self._feature_detector = DelfFeature2D(
                num_features=self.num_features, score_threshold=20)
            self.scale_factor = self._feature_detector.scale_factor
            #self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.CONTEXTDESC:
            self.set_sift_parameters()
            self.need_color_image = True
            # self.num_levels = 1 # force              # computed internally by SIFT method
            self._feature_detector = ContextDescFeature2D(
                num_features=self.num_features)
            #self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.LFNET:
            self.need_color_image = True
            # self.num_levels = 1 # force
            self._feature_detector = LfNetFeature2D(
                num_features=self.num_features)
            #self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.R2D2:
            self.need_color_image = True
            # self.num_levels = - # internally recomputed
            self._feature_detector = R2d2Feature2D(
                num_features=self.num_features)
            self.scale_factor = self._feature_detector.scale_f
            self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.KEYNET:
            # self.num_levels = - # internally recomputed
            self._feature_detector = KeyNetDescFeature2D(
                num_features=self.num_features)
            self.num_features = self._feature_detector.num_features
            self.num_levels = self._feature_detector.num_levels
            self.scale_factor = self._feature_detector.scale_factor
            self.keypoint_filter_type = KeyPointFilterTypes.NONE
            #
            #
        elif self.detector_type == FeatureDetectorTypes.DISK:
            self.num_levels = 1  # force
            self.need_color_image = True
            self._feature_detector = DiskFeature2D(
                num_features=self.num_features)
            #
            #
        else:
            raise ValueError("Unknown feature detector %s" %
                             self.detector_type)

        if self.need_nms:
            self.keypoint_filter_type = self.keypoint_nms_filter_type

        if self.use_bock_adaptor:
            self.orb_params['edgeThreshold'] = 0

        # --------------------------------------------- #
        # init descriptor
        # --------------------------------------------- #
        if self.is_detector_equal_to_descriptor:
            Printer.green(
                'using same detector and descriptor object: ', self.detector_type.name)
            self._feature_descriptor = self._feature_detector
        else:
            # detector and descriptors are different
            self.num_levels_descriptor = self.num_levels
            if self.use_pyramid_adaptor:
                # NOT VALID ANYMORE -> if there is a pyramid adaptor, the descriptor does not need to rescale the images which are rescaled by the pyramid adaptor itself
                #self.orb_params['nlevels'] = 1
                # self.num_levels_descriptor = 1 #self.num_levels
                pass
            # actual descriptor initialization
            if self.descriptor_type == FeatureDescriptorTypes.SIFT or self.descriptor_type == FeatureDescriptorTypes.ROOT_SIFT:
                sift = self.SIFT_create(nOctaveLayers=3)
                if self.descriptor_type == FeatureDescriptorTypes.ROOT_SIFT:
                    self._feature_descriptor = RootSIFTFeature2D(sift)
                else:
                    self._feature_descriptor = sift
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.SURF:
                self.oriented_features = True  # SURF computes the keypoint orientation
                self._feature_descriptor = self.SURF_create(
                    nOctaves=self.num_levels_descriptor, nOctaveLayers=3)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.ORB:
                self._feature_descriptor = self.ORB_create(**self.orb_params)
                # self.oriented_features = False   # N.B: ORB descriptor does not compute orientation on its own
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.ORB2:
                self._feature_descriptor = self.ORB_create(**self.orb_params)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.BRISK:
                self.oriented_features = True  # BRISK computes the keypoint orientation
                self._feature_descriptor = self.BRISK_create(
                    octaves=self.num_levels_descriptor)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.KAZE:
                if not self.is_detector_equal_to_descriptor:
                    # https://kyamagu.github.io/mexopencv/matlab/AKAZE.html
                    Printer.red(
                        'WARNING: KAZE descriptors can only be used with KAZE or AKAZE keypoints.')
                self._feature_descriptor = self.KAZE_create(
                    nOctaves=self.num_levels_descriptor)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.AKAZE:
                if not self.is_detector_equal_to_descriptor:
                    # https://kyamagu.github.io/mexopencv/matlab/AKAZE.html
                    Printer.red(
                        'WARNING: AKAZE descriptors can only be used with KAZE or AKAZE keypoints.')
                self._feature_descriptor = self.AKAZE_create(
                    nOctaves=self.num_levels_descriptor)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.FREAK:
                self.oriented_features = True  # FREAK computes the keypoint orientation
                self._feature_descriptor = self.FREAK_create(
                    nOctaves=self.num_levels_descriptor)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.SUPERPOINT:
                if self.detector_type != FeatureDetectorTypes.SUPERPOINT:
                    raise ValueError(
                        "You cannot use SUPERPOINT descriptor without SUPERPOINT detector!\nPlease, select SUPERPOINT as both descriptor and detector!")
                # reuse the same SuperPointDector object
                self._feature_descriptor = self._feature_detector
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.TFEAT:
                self._feature_descriptor = TfeatFeature2D()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.BOOST_DESC:
                # below a proper keypoint size scale factor is set depending on the used detector
                self.do_keypoints_size_rescaling = False
                boost_des_keypoint_size_scale_factor = 1.5
                # from https://docs.opencv.org/3.4.2/d1/dfd/classcv_1_1xfeatures2d_1_1BoostDesc.html#details
                # scale_factor:	adjust the sampling window of detected keypoints 6.25f is default and fits for KAZE, SURF
                #               detected keypoints window ratio 6.75f should be the scale for SIFT
                #               detected keypoints window ratio 5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK
                #               keypoints window ratio 0.75f should be the scale for ORB
                #               keypoints ratio 1.50f was the default in original implementation
                if self.detector_type in [FeatureDetectorTypes.KAZE, FeatureDetectorTypes.SURF]:
                    boost_des_keypoint_size_scale_factor = 6.25
                elif self.detector_type == FeatureDetectorTypes.SIFT:
                    boost_des_keypoint_size_scale_factor = 6.75
                elif self.detector_type in [FeatureDetectorTypes.AKAZE, FeatureDetectorTypes.AGAST, FeatureDetectorTypes.FAST, FeatureDetectorTypes.BRISK]:
                    boost_des_keypoint_size_scale_factor = 5.0
                elif self.detector_type == FeatureDetectorTypes.ORB:
                    boost_des_keypoint_size_scale_factor = 0.75
                self._feature_descriptor = self.BoostDesc_create(
                    scale_factor=boost_des_keypoint_size_scale_factor)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.DAISY:
                self._feature_descriptor = self.DAISY_create()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.LATCH:
                self._feature_descriptor = self.LATCH_create()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.LUCID:
                self._feature_descriptor = self.LUCID_create(lucid_kernel=1,  # =1
                                                             blur_kernel=3)  # =2
                self.need_color_image = True
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.VGG:
                self._feature_descriptor = self.VGG_create()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.HARDNET:
                self._feature_descriptor = HardnetFeature2D(do_cuda=True)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.GEODESC:
                self._feature_descriptor = GeodescFeature2D()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.SOSNET:
                self._feature_descriptor = SosnetFeature2D()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.L2NET:
                # self._feature_descriptor = L2NetKerasFeature2D()    # keras-tf version
                self._feature_descriptor = L2NetFeature2D()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.LOGPOLAR:
                self._feature_descriptor = LogpolarFeature2D()
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.D2NET:
                self.need_color_image = True
                if self.detector_type != FeatureDetectorTypes.D2NET:
                    raise ValueError(
                        "You cannot use D2NET descriptor without D2NET detector!\nPlease, select D2NET as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.DELF:
                self.need_color_image = True
                if self.detector_type != FeatureDetectorTypes.DELF:
                    raise ValueError(
                        "You cannot use DELF descriptor without DELF detector!\nPlease, select DELF as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.CONTEXTDESC:
                self.need_color_image = True
                if self.detector_type != FeatureDetectorTypes.CONTEXTDESC:
                    raise ValueError(
                        "You cannot use CONTEXTDESC descriptor without CONTEXTDESC detector!\nPlease, select CONTEXTDESC as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.LFNET:
                self.need_color_image = True
                if self.detector_type != FeatureDetectorTypes.LFNET:
                    raise ValueError(
                        "You cannot use LFNET descriptor without LFNET detector!\nPlease, select LFNET as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.R2D2:
                self.oriented_features = False
                self.need_color_image = True
                if self.detector_type != FeatureDetectorTypes.R2D2:
                    raise ValueError(
                        "You cannot use R2D2 descriptor without R2D2 detector!\nPlease, select R2D2 as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.KEYNET:
                self.oriented_features = False
                if self.detector_type != FeatureDetectorTypes.KEYNET:
                    raise ValueError(
                        "You cannot use KEYNET internal descriptor without KEYNET detector!\nPlease, select KEYNET as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.BEBLID:
                # https://docs.opencv.org/master/d7/d99/classcv_1_1xfeatures2d_1_1BEBLID.html
                BEBLID_SIZE_256_BITS = 101
                BEBLID_scale_factor = 1.0   # it depends on the used detector https://docs.opencv.org/master/d7/d99/classcv_1_1xfeatures2d_1_1BEBLID.html#a38997aa059977abf6a2d6bf462d50de0a7b2a1e106c93d76cdfe5cef053277a04
                # TODO: adapt BEBLID scale factor to actual used detector
                #       1.0 is OK for ORB2 detector
                self._feature_descriptor = self.BEBLID_create(
                    BEBLID_scale_factor, BEBLID_SIZE_256_BITS)
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.DISK:
                self.oriented_features = False
                if self.detector_type != FeatureDetectorTypes.DISK:
                    raise ValueError(
                        "You cannot use DISK internal descriptor without DISK detector!\nPlease, select DISK as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector  # reuse detector object
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.NONE:
                self._feature_descriptor = None
            else:
                raise ValueError("Unknown feature descriptor %s" %
                                 self.descriptor_type)

        # --------------------------------------------- #
        # init from FeatureInfo
        # --------------------------------------------- #

        # get and set norm type
        try:
            self.norm_type = FeatureInfo.norm_type[self.descriptor_type]
        except:
            Printer.red('You did not set the norm type for: ',
                        self.descriptor_type.name)
            raise ValueError(
                "Unmanaged norm type for feature descriptor %s" % self.descriptor_type.name)

        # set descriptor distance functions
        if self.norm_type == cv2.NORM_HAMMING:
            self.descriptor_distance = hamming_distance
            self.descriptor_distances = hamming_distances
        if self.norm_type == cv2.NORM_L2:
            self.descriptor_distance = l2_distance
            self.descriptor_distances = l2_distances

         # get and set reference max descriptor distance
        try:
            kMaxDescriptorDistance = FeatureInfo.max_descriptor_distance[self.descriptor_type]
        except:
            Printer.red(
                'You did not set the reference max descriptor distance for: ', self.descriptor_type.name)
            raise ValueError(
                "Unmanaged max descriptor distance for feature descriptor %s" % self.descriptor_type.name)
        
        kMaxDescriptorDistanceSearchEpipolar = kMaxDescriptorDistance

        # --------------------------------------------- #
        # other required initializations
        # --------------------------------------------- #

        if not self.oriented_features:
            Printer.orange('WARNING: using NON-ORIENTED features: ', self.detector_type.name,
                           '-', self.descriptor_type.name, ' (i.e. kp.angle=0)')

        if self.is_detector_equal_to_descriptor and \
            (self.detector_type == FeatureDetectorTypes.SIFT or
             self.detector_type == FeatureDetectorTypes.ROOT_SIFT or
             self.detector_type == FeatureDetectorTypes.CONTEXTDESC):
            self.init_sigma_levels_sift()
        else:
            self.init_sigma_levels()

        if self.use_bock_adaptor:
            self.block_adaptor = BlockAdaptor(
                self._feature_detector, self._feature_descriptor)

        if self.use_pyramid_adaptor:
            self.pyramid_params = dict(detector=self._feature_detector,
                                       descriptor=self._feature_descriptor,
                                       num_features=self.num_features,
                                       num_levels=self.num_levels,
                                       scale_factor=self.scale_factor,
                                       sigma0=self.sigma_level0,
                                       first_level=self.first_level,
                                       pyramid_type=self.pyramid_type,
                                       use_block_adaptor=self.use_bock_adaptor,
                                       do_parallel=self.pyramid_do_parallel,
                                       do_sat_features_per_level=self.do_sat_features_per_level)
            self.pyramid_adaptor = PyramidAdaptor(**self.pyramid_params)

    def set_sift_parameters(self):
        # N.B.: The number of SIFT octaves is automatically computed from the image resolution,
        #       here we can set the number of layers in each octave.
        #       from https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
        # self.intra_layer_factor = 1.2599   # num layers = nOctaves*nOctaveLayers  scale=2^(1/nOctaveLayers) = 1.2599
        self.scale_factor = 2              # force scale factor = 2 between octaves
        # https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L118
        self.sigma_level0 = 1.6
        # from https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
        # https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L731
        self.first_level = -1

    # initialize scale factors, sigmas for each octave level;
    # these are used for managing image pyramids and weighting (information matrix) reprojection error terms in the optimization

    def init_sigma_levels(self):
        print('num_levels: ', self.num_levels)
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.inv_scale_factor = 1./self.scale_factor
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale_factor)

        self.scale_factors[0] = 1.0
        self.level_sigmas2[0] = self.sigma_level0*self.sigma_level0
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            self.level_sigmas2[i] = self.scale_factors[i] * \
                self.scale_factors[i]*self.level_sigmas2[0]
            self.level_sigmas[i] = math.sqrt(self.level_sigmas2[i])
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0/self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0/self.level_sigmas2[i]
        #print('self.scale_factor: ', self.scale_factor)
        #print('self.scale_factors: ', self.scale_factors)
        #print('self.level_sigmas: ', self.level_sigmas)
        #print('self.inv_scale_factors: ', self.inv_scale_factors)

    # initialize scale factors, sigmas for each octave level;
    # these are used for managing image pyramids and weighting (information matrix) reprojection error terms in the optimization;
    # this method can be used only when the following mapping is adopted for SIFT:
    #   keypoint.octave = (unpacked_octave+1)*3+unpacked_layer  where S=3 is the number of levels per octave

    def init_sigma_levels_sift(self):
        print('initializing SIFT sigma levels')
        print('num_levels: ', self.num_levels)
        # we map: level=keypoint.octave = (unpacked_octave+1)*3+unpacked_layer  where S=3 is the number of scales per octave
        self.num_levels = 3*self.num_levels + 3
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        #print('num_levels: ', num_levels)
        # N.B: if we adopt the mapping: keypoint.octave = (unpacked_octave+1)*3+unpacked_layer
        # then we can consider a new virtual scale_factor = 2^(1/3) (used between two contiguous layers of the same octave)
        print('original scale factor: ', self.scale_factor)
        self.scale_factor = math.pow(2, 1./3)
        self.inv_scale_factor = 1./self.scale_factor
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale_factor)

        # https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L118
        self.sigma_level0 = 1.6
        # from https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
        sigma_level02 = self.sigma_level0*self.sigma_level0

        # N.B.: these are used only when recursive filtering is applied: see https://www.vlfeat.org/api/sift.html#sift-tech-ss
        #sift_init_sigma = 0.5
        #sift_init_sigma2 = 0.25

        # see also https://www.vlfeat.org/api/sift.html
        self.scale_factors[0] = 1.0
        # -4*sift_init_sigma2  N.B.: this is an absolute sigma,
        self.level_sigmas2[0] = sigma_level02
        # not a delta_sigma used for incrementally filtering contiguos layers => we must not subtract (4*sift_init_sigma2)
        # https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L197
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            # https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L224
            self.level_sigmas2[i] = self.scale_factors[i] * \
                self.scale_factors[i]*sigma_level02
            self.level_sigmas[i] = math.sqrt(self.level_sigmas2[i])
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0/self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0/self.level_sigmas2[i]
        #print('self.scale_factor: ', self.scale_factor)
        #print('self.scale_factors: ', self.scale_factors)
        #print('self.level_sigmas: ', self.level_sigmas)
        #print('self.inv_scale_factors: ', self.inv_scale_factors)

    # filter matches by using
    # Non-Maxima Suppression (NMS) based on kd-trees
    # or SSC NMS (https://github.com/BAILOOL/ANMS-Codes)
    # or SAT (get features with best responses)
    # or OCTREE_NMS (implemented in ORBSLAM2, distribution of features in a quad-tree)

    def filter_keypoints(self, type, frame, kps, des=None):
        filter_name = type.name
        if type == KeyPointFilterTypes.NONE:
            pass
        elif type == KeyPointFilterTypes.KDT_NMS:
            kps, des = kdt_nms(kps, des, self.num_features)
        elif type == KeyPointFilterTypes.SSC_NMS:
            kps, des = ssc_nms(
                kps, des, frame.shape[1], frame.shape[0], self.num_features)
        elif type == KeyPointFilterTypes.OCTREE_NMS:
            if des is not None:
                raise ValueError(
                    'at the present time, you cannot use OCTREE_NMS with descriptors')
            kps = octree_nms(frame, kps, self.num_features)
        elif type == KeyPointFilterTypes.GRID_NMS:
            kps, des, _ = grid_nms(
                kps, des, frame.shape[0], frame.shape[1], self.num_features, dist_thresh=4)
        elif type == KeyPointFilterTypes.SAT:
            if len(kps) > self.num_features:
                kps, des = sat_num_features(kps, des, self.num_features)
        else:
            raise ValueError("Unknown match-filter type")
        return kps, des, filter_name


    def rescale_keypoint_size(self, kps):
        # if keypoints are FAST, etc. then rescale their small sizes
        # in order to let descriptors compute an encoded representation with a decent patch size
        scale = 1
        doit = False
        if self.detector_type == FeatureDetectorTypes.FAST:
            scale = kFASTKeyPointSizeRescaleFactor
            doit = True
        elif self.detector_type == FeatureDetectorTypes.AGAST:
            scale = kAGASTKeyPointSizeRescaleFactor
            doit = True
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI or self.detector_type == FeatureDetectorTypes.GFTT:
            scale = kShiTomasiKeyPointSizeRescaleFactor
            doit = True
        if doit:
            for kp in kps:
                kp.size *= scale

    # detect keypoints without computing their descriptors
    # out: kps (array of cv2.KeyPoint)
    def detect(self, frame, mask=None, filter=True):
        # check if we have to convert to gray image
        if not self.need_color_image and frame.ndim > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.use_pyramid_adaptor:
            # detection with pyramid adaptor (it can optionally include a block adaptor per level)
            kps = self.pyramid_adaptor.detect(frame, mask)
        elif self.use_bock_adaptor:
            # detection with block adaptor
            kps = self.block_adaptor.detect(frame, mask)
        else:
            # standard detection
            kps = self._feature_detector.detect(frame, mask)
        # filter keypoints
        filter_name = 'NONE'
        if filter:
            kps, _, filter_name = self.filter_keypoints(self.keypoint_filter_type, frame, kps)
        
        # if keypoints are FAST, etc. give them a decent size in order to properly compute the descriptors
        if self.do_keypoints_size_rescaling:
            self.rescale_keypoint_size(kps)
        
        if kDrawOriginalExtractedFeatures:  # draw the original features
            imgDraw = cv2.drawKeypoints(
                frame, kps, None, color=(0, 255, 0), flags=0)
            cv2.imshow('detected keypoints', imgDraw)
        
        if kVerbose:
            print('detector:', self.detector_type.name, ', #features:',
                  len(kps), ', [kp-filter:', filter_name, ']')
        return kps

    # compute the descriptors once given the keypoints

    def compute(self, frame, kps, filter=True):
        if not self.need_color_image and frame.ndim > 2:     # check if we have to convert to gray image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kps, des = self._feature_descriptor.compute(
            frame, kps)  # then, compute descriptors
        # filter keypoints
        filter_name = 'NONE'
        if filter:
            kps, des, filter_name = self.filter_keypoints(
                self.keypoint_filter_type, frame, kps, des)
        if kVerbose:
            print('descriptor:', self.descriptor_type.name,
                  ', #features:', len(kps), ', [kp-filter:', filter_name, ']')
        return kps, des

    # detect keypoints and their descriptors
    # out: kps, des

    def detectAndCompute(self, frame, mask=None, filter=True):
        if not self.need_color_image and frame.ndim > 2:     # check if we have to convert to gray image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.use_pyramid_adaptor:
            # detectAndCompute with pyramid adaptor (it can optionally include a block adaptor per level)
            if self.force_multiscale_detect_and_compute:
                # force detectAndCompute on each level instead of first {detect() on each level} and then {compute() on resulting detected keypoints one time}
                kps, des = self.pyramid_adaptor.detectAndCompute(frame, mask)
            #
            else:
                # first, detect by using adaptor on the different pyramid levels
                kps = self.detect(frame, mask, filter=True)
                # then, separately compute the descriptors on detected keypoints (one time)
                kps, des = self.compute(frame, kps, filter=False)
                filter = False  # disable keypoint filtering since we already applied it for detection
        elif self.use_bock_adaptor:
            # detectAndCompute with block adaptor (force detect/compute on each block)
            #
            #kps, des = self.block_adaptor.detectAndCompute(frame, mask)
            #
            # first, detect by using adaptor
            kps = self.detect(frame, mask, filter=True)
            # then, separately compute the descriptors
            kps, des = self.compute(frame, kps, filter=False)
            filter = False  # disable keypoint filtering since we already applied it for detection
        else:
            # standard detectAndCompute
            if self.is_detector_equal_to_descriptor:
                # detector = descriptor => call them together with detectAndCompute() method
                kps, des = self._feature_detector.detectAndCompute(frame, mask)
                if kVerbose:
                    print('detector:', self.detector_type.name,
                          ', #features:', len(kps))
                    print('descriptor:', self.descriptor_type.name,
                          ', #features:', len(kps))
            else:
                # detector and descriptor are different => call them separately
                # 1. first, detect keypoint locations
                kps = self.detect(frame, mask, filter=False)
                # 2. then, compute descriptors
                kps, des = self._feature_descriptor.compute(frame, kps)
                if kVerbose:
                    # print('detector: ', self.detector_type.name, ', #features: ', len(kps))
                    print('descriptor: ', self.descriptor_type.name,
                          ', #features: ', len(kps))
        # filter keypoints
        filter_name = 'NONE'
        if filter:
            kps, des, filter_name = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)
        
        if self.detector_type == FeatureDetectorTypes.SIFT or \
           self.detector_type == FeatureDetectorTypes.ROOT_SIFT or \
           self.detector_type == FeatureDetectorTypes.CONTEXTDESC:
            unpackSiftOctaveKps(kps, method=UnpackOctaveMethod.INTRAL_LAYERS)
        
        if kVerbose:
            print('detector:', self.detector_type.name, ', descriptor:', self.descriptor_type.name,
                  ', #features:', len(kps), ' (#ref:', self.num_features, '), [kp-filter:', filter_name, ']')
        self.debug_print(kps)
        return kps, des

    def debug_print(self, kps):
        if False:
            # raw print of all keypoints
            for k in kps:
                print("response: ", k.response, "\t, size: ", k.size,
                      "\t, octave: ", k.octave, "\t, angle: ", k.angle)
        if False:
            # generate a rough histogram for keypoint sizes
            kps_sizes = [kp.size for kp in kps]
            kps_sizes_histogram = np.histogram(kps_sizes, bins=10)
            print('size-histogram: \n',
                  list(zip(kps_sizes_histogram[1], kps_sizes_histogram[0])))
            # generate histogram at level 0
            kps_sizes = [kp.size for kp in kps if kp.octave == 1]
            kps_sizes_histogram = np.histogram(kps_sizes, bins=10)
            print('size-histogram at level 0: \n',
                  list(zip(kps_sizes_histogram[1], kps_sizes_histogram[0])))
        if False:
            # count points for each octave => generate an octave histogram
            kps_octaves = [k.octave for k in kps]
            kps_octaves = Counter(kps_octaves)
            print('levels-histogram: ', kps_octaves.most_common(12))


# BlockAdaptor divides the image in row_divs x col_divs cells and extracts features in each of these cells
class BlockAdaptor(object):
    def __init__(self,
                 detector,
                 descriptor=None,
                 row_divs=1,
                 col_divs=1,
                 do_parallel=1):
        self.detector = detector
        self.descriptor = descriptor
        self.row_divs = row_divs
        self.col_divs = col_divs
        self.do_parallel = do_parallel  # do parallel computations
        self.is_detector_equal_to_descriptor = (
            self.detector == self.descriptor)

    def detect(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1:
            return self.detector.detect(frame, mask)
        else:
            if kVerbose:
                print('BlockAdaptor ', self.row_divs, 'x', self.col_divs)
            block_generator = img_mask_blocks(
                frame, mask, self.row_divs, self.col_divs)
            kps_all = []  # list are thread-safe

            def detect_block(b_m_i_j):
                b, m, i, j = b_m_i_j
                if kVerbose and False:
                    print('BlockAdaptor  in block (', i, ',', j, ')')
                kps = self.detector.detect(b, mask=m)
                # print('adaptor: detected #features: ', len(kps), ' in block (',i,',',j,')')
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)
                    #print('kp.pt after: ', kp.pt)
                kps_all.extend(kps)

            if not self.do_parallel:
                # process the blocks sequentially
                for b, m, i, j in block_generator:
                    detect_block((b, m, i, j))
            else:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # automatic join() at the end of the `width` block
                    executor.map(detect_block, block_generator)
            return np.array(kps_all)

    def detectAndCompute(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1:
            return self.detector.detectAndCompute(frame, mask)
        else:
            if kVerbose:
                print('BlockAdaptor ', self.row_divs, 'x', self.col_divs)
            block_generator = img_mask_blocks(
                frame, mask, self.row_divs, self.col_divs)
            kps_all = []
            des_all = []
            kps_des_map = {}  # (i,j) -> (kps,des)

            def detect_and_compute_block(b_m_i_j):
                b, m, i, j = b_m_i_j
                if kVerbose and False:
                    print('BlockAdaptor  in block (', i, ',', j, ')')
                if self.is_detector_equal_to_descriptor:
                    kps, des = self.detector.detectAndCompute(b, mask=m)
                else:
                    kps = self.detector.detect(b, mask=m)
                    kps, des = self.descriptor.compute(b, kps)
                    # print('adaptor: detected #features: ', len(kps), ' in block (',i,',',j,')')
                # transform the points
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)
                    #print('kp.pt after: ', kp.pt)
                kps_des_map[(i, j)] = (kps, des)

            if not self.do_parallel:
                # process the blocks sequentially
                for b, m, i, j in block_generator:
                    detect_and_compute_block((b, m, i, j))
            else:
                with ThreadPoolExecutor(max_workers=kBlockAdaptorMaxNumWorkers) as executor:
                    # automatic join() at the end of the `width` block
                    executor.map(detect_and_compute_block, block_generator)

            # now merge the computed results
            for ij, (kps, des) in kps_des_map.items():
                kps_all.extend(kps)
                if des is not None and len(des) > 0:
                    if len(des_all) > 0:
                        des_all = np.vstack([des_all, des])
                    else:
                        des_all = des
            return np.array(kps_all), np.array(des_all)


# PyramidAdaptor generate a pyramid of num_levels images and extracts features in each of these images
# TODO: check if a point on one level 'overlaps' with a point on other levels or add such option (DONE by FeatureManager.kdt_nms() )
class PyramidAdaptor(object):
    def __init__(self,
                 detector,
                 descriptor=None,
                 num_features=2000,
                 num_levels=4,
                 scale_factor=1.2,
                 sigma0=1.0,     # N.B.: SIFT use 1.6 for this value
                 first_level=0,
                 pyramid_type=1,
                 use_block_adaptor=False,
                 do_parallel=True,
                 do_sat_features_per_level=False):
        self.detector = detector
        self.descriptor = descriptor
        self.num_features = num_features
        self.is_detector_equal_to_descriptor = (
            self.detector == self.descriptor)
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.inv_scale_factor = 1./scale_factor
        self.sigma0 = sigma0
        self.first_level = first_level
        self.pyramid_type = pyramid_type
        self.use_block_adaptor = use_block_adaptor
        self.do_parallel = do_parallel   # do parallel computations
        # saturate number of features for each level
        self.do_sat_features_per_level = do_sat_features_per_level

        self.pyramid = Pyramid(num_levels=num_levels,
                               scale_factor=scale_factor,
                               sigma0=sigma0,
                               first_level=first_level,
                               pyramid_type=pyramid_type)
        self.initSigmaLevels()

        self.block_adaptor = None
        if self.use_block_adaptor:
            self.block_adaptor = BlockAdaptor(
                self.detector, self.descriptor, row_divs=1, col_divs=kAdaptorNumColDivs, do_parallel=False)

    def initSigmaLevels(self):
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.scale_factors = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.scale_factors[0] = 1.0

        # compute desired number of features per level (by using the scale factor)
        self.num_features_per_level = np.zeros(num_levels, dtype=np.int)
        num_desired_features_per_level = self.num_features * \
            (1 - self.inv_scale_factor) / \
            (1 - math.pow(self.inv_scale_factor, self.num_levels))
        sum_num_features = 0
        for level in range(self.num_levels-1):
            self.num_features_per_level[level] = int(
                round(num_desired_features_per_level))
            sum_num_features += self.num_features_per_level[level]
            num_desired_features_per_level *= self.inv_scale_factor
        self.num_features_per_level[self.num_levels -
                                    1] = max(self.num_features - sum_num_features, 0)
        # print('num_features_per_level:',self.num_features_per_level)

        if self.first_level == -1:
            self.scale_factors[0] = 1.0/self.scale_factor
        self.inv_scale_factors[0] = 1.0/self.scale_factors[0]
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            self.inv_scale_factors[i] = 1.0/self.scale_factors[i]
        #print('self.inv_scale_factors: ', self.inv_scale_factors)

    # detect on 'unfiltered' pyramid images ('unfiltered' meanining depends on the selected pyramid type)

    def detect(self, frame, mask=None):
        if self.num_levels == 1:
            return self.detector.detect(frame, mask)
        else:
            # TODO: manage mask
            if kVerbose:
                print('PyramidAdaptor #levels:', self.num_levels, '(from', self.first_level, '), scale_factor:',
                      self.scale_factor, ', sigma0:', self.sigma0, ', type:', self.pyramid_type.name)
            self.pyramid.compute(frame)
            kps_all = []  # list are thread-safe

            def detect_level(scale, pyr_cur, i):
                kps = []
                if self.block_adaptor is None:
                    kps = self.detector.detect(pyr_cur)
                else:
                    kps = self.block_adaptor.detect(pyr_cur)
                if kVerbose and False:
                    print("PyramidAdaptor - level", i,
                          ", shape: ", pyr_cur.shape)
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0]*scale, kp.pt[1]*scale)
                    kp.size = kp.size*scale
                    kp.octave = i
                    #print('kp: ', kp.pt, kp.octave)
                if self.do_sat_features_per_level:
                    kps, _ = sat_num_features(
                        kps, None, self.num_features_per_level[i])  # experimental
                kps_all.extend(kps)

            if not self.do_parallel:
                #print('sequential computations')
                # process the blocks sequentially
                for i in range(0, self.num_levels):
                    scale = self.scale_factors[i]
                    pyr_cur = self.pyramid.imgs[i]
                    detect_level(scale, pyr_cur, i)
            else:
                #print('parallel computations')
                futures = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for i in range(0, self.num_levels):
                        scale = self.scale_factors[i]
                        pyr_cur = self.pyramid.imgs[i]
                        futures.append(executor.submit(
                            detect_level, scale, pyr_cur, i))
                    wait(futures)  # wait all the task are completed

            return np.array(kps_all)

    # detect on 'unfiltered' pyramid images ('unfiltered' meanining depends on the selected pyramid type)
    # compute descriptors on 'filtered' pyramid images ('filtered' meanining depends on the selected pyramid type)

    def detectAndCompute(self, frame, mask=None):
        if self.num_levels == 1:
            return self.detector.detectAndCompute(frame, mask)
        else:
            if kVerbose:
                print('PyramidAdaptor [dc] #levels:', self.num_levels, '(from', self.first_level, '), scale_factor:',
                      self.scale_factor, ', sigma0:', self.sigma0, ', type:', self.pyramid_type.name)
            self.pyramid.compute(frame)
            kps_all = []
            des_all = []
            kps_des_map = {}  # i -> (kps,des)

            def detect_and_compute_level(scale, pyr_cur, pyr_cur_filtered, N, i):
                kps = []
                if self.block_adaptor is None:
                    #kps, des = self.detector.detectAndCompute(pyr_cur)
                    if self.is_detector_equal_to_descriptor:
                        kps, des = self.detector.detectAndCompute(pyr_cur)
                    else:
                        kps = self.detector.detect(pyr_cur)
                        #print('description of filtered')
                        kps, des = self.descriptor.compute(
                            pyr_cur_filtered, kps)
                else:
                    kps, des = self.block_adaptor.detectAndCompute(pyr_cur)
                if kVerbose and False:
                    print("PyramidAdaptor - level", i,
                          ", shape: ", pyr_cur.shape)
                for kp in kps:
                    #print('before: kp.pt:', kp.pt,', size:',kp.size,', octave:',kp.octave,', angle:',kp.angle)
                    kp.pt = (kp.pt[0]*scale, kp.pt[1]*scale)
                    kp.size = kp.size*scale
                    kp.octave = i
                    #print('after: kp.pt:', kp.pt,', size:',kp.size,', octave:',kp.octave,', angle:',kp.angle)
                if self.do_sat_features_per_level:
                    kps, des = sat_num_features(kps, des, N)  # experimental
                kps_des_map[i] = (kps, des)

            if not self.do_parallel:
                #print('sequential computations')
                # process the blocks sequentially
                for i in range(0, self.num_levels):
                    scale = self.scale_factors[i]
                    pyr_cur = self.pyramid.imgs[i]
                    pyr_cur_filtered = self.pyramid.imgs_filtered[i]
                    detect_and_compute_level(
                        scale, pyr_cur, pyr_cur_filtered, self.num_features_per_level[i], i)
            else:
                #print('parallel computations')
                futures = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for i in range(0, self.num_levels):
                        scale = self.scale_factors[i]
                        pyr_cur = self.pyramid.imgs[i]
                        pyr_cur_filtered = self.pyramid.imgs_filtered[i]
                        futures.append(executor.submit(detect_and_compute_level, scale,
                                       pyr_cur, pyr_cur_filtered, self.num_features_per_level[i], i))
                    wait(futures)  # wait all the task are completed

            # now merge the computed results
            for i, (kps, des) in kps_des_map.items():
                kps_all.extend(kps)
                if des is not None and len(des) > 0:
                    if len(des_all) > 0:
                        des_all = np.vstack([des_all, des])
                    else:
                        des_all = des
            return np.array(kps_all), np.array(des_all)

kNumLevelsInitSigma = 20

# pyramid types
class PyramidType(Enum):
    # only resize, do NOT filter (N.B.: filters are typically applied for obtaining a useful antialiasing effect)
    RESIZE = 0
    # both Pyramid.imgs and Pyramid.imgs_filtered contain unfiltered resized images
    # compute separated resized images and filtered images: first resize then filter (typically used by ORB)
    RESIZE_AND_FILTER = 1
    # Pyramid.imgs contains (unfiltered) resized images, and Pyramid.imgs_filtered contain filtered resized images
    # compute images in the scale-space: first filter (with appropriate sigmas) than resize, see  https://www.vlfeat.org/api/sift.html#sift-tech-ss  (used by SIFT, SURF, etc...)
    GAUSS_PYRAMID = 2
    # both Pyramid.imgs and Pyramid.imgs_filtered contain filtered images in the scale space


# PyramidAdaptor generate a pyramid of num_levels images and extracts features in each of these images
class Pyramid(object):
    def __init__(self, num_levels=4, scale_factor=1.2,
                 sigma0=1.0,     # N.B.: SIFT use 1.6 for this value
                 first_level=0,  # 0: start from input image; -1: start from up-scaled image*scale_factor
                 pyramid_type=PyramidType.RESIZE):
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.sigma0 = sigma0
        self.first_level = first_level
        self.pyramid_type = pyramid_type

        self.imgs = []
        self.imgs_filtered = []
        self.base_img = None

        self.scale_factors = None
        self.inv_scale_factors = None
        self.initSigmaLevels()

    def initSigmaLevels(self):
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.scale_factors = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.scale_factors[0] = 1.0
        self.inv_scale_factors[0] = 1.0/self.scale_factors[0]
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            self.inv_scale_factors[i] = 1.0/self.scale_factors[i]
        #print('self.inv_scale_factors: ', self.inv_scale_factors)

    def compute(self, frame):
        if self.first_level == -1:
            # replace the image with the new level -1 (up-resized image)
            frame = self.createBaseImg(frame)
        if self.pyramid_type == PyramidType.RESIZE:
            return self.computeResize(frame)
        elif self.pyramid_type == PyramidType.RESIZE_AND_FILTER:
            return self.computeResizeAndFilter(frame)
        elif self.pyramid_type == PyramidType.GAUSS_PYRAMID:
            return self.computeGauss(frame)
        else:
            Printer.orange('Pyramid - unknown type')
            return self.computeResizePyramid(frame)

    def createBaseImg(self, frame):
        # 0.5 is the base sigma from https://www.vlfeat.org/api/sift.html#sift-tech-ss
        sigma_init = 0.5
        # see https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L197
        delta_sigma = math.sqrt(max(
            self.sigma0*self.sigma0 - (sigma_init*sigma_init*self.scale_factor*self.scale_factor), 0.01))
        frame_upscaled = cv2.resize(
            frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if self.pyramid_type == PyramidType.GAUSS_PYRAMID:
            return cv2.GaussianBlur(frame_upscaled, ksize=(0, 0), sigmaX=delta_sigma)
        else:
            return frame_upscaled

    # only resize, do not filter

    def computeResize(self, frame):
        inv_scale = 1./self.scale_factor
        self.imgs = []
        self.imgs_filtered = []
        pyr_cur = frame
        for i in range(0, self.num_levels):
            self.imgs.append(pyr_cur)
            self.imgs_filtered.append(pyr_cur)
            if i < self.num_levels-1:
                # resize the unfiltered frame
                pyr_down = cv2.resize(
                    pyr_cur, (0, 0), fx=inv_scale, fy=inv_scale)
                pyr_cur = pyr_down

    # keep separated resized images and filtered images: first resize than filter with constant sigma

    def computeResizeAndFilter(self, frame):
        inv_scale = 1./self.scale_factor
        filter_sigmaX = 2  # setting used for computing ORB descriptors
        ksize = (5, 5)
        self.imgs = []
        self.imgs_filtered = []
        pyr_cur = frame
        for i in range(0, self.num_levels):
            filtered = cv2.GaussianBlur(pyr_cur, ksize, sigmaX=filter_sigmaX)
            # self.imgs contain resized image
            self.imgs.append(pyr_cur)
            # self.imgs_filtered contain filtered images
            self.imgs_filtered.append(filtered)
            if i < self.num_levels-1:
                # resize the unfiltered frame
                pyr_down = cv2.resize(
                    pyr_cur, (0, 0), fx=inv_scale, fy=inv_scale)
                pyr_cur = pyr_down

    # compute images in the scale space: first filter (with appropriate sigmas) than resize

    def computeGauss(self, frame):
        inv_scale = 1./self.scale_factor

        # from https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L212
        # \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
        # original image with nominal sigma=0.5  <=from https://www.vlfeat.org/api/sift.html#sift-tech-ss
        sigma_nominal = 0.5
        sigma0 = self.sigma0  # N.B.: SIFT use 1.6 for this value
        sigma_prev = sigma_nominal

        self.imgs = []
        self.imgs_filtered = []

        pyr_cur = frame

        for i in range(0, self.num_levels):
            if i == 0 and self.first_level == -1:
                sigma_prev = sigma0
                filtered = frame
            else:
                sigma_total = self.scale_factors[i] * sigma0
                # this the DELTA-SIGMA according to \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
                sigma_cur = math.sqrt(
                    sigma_total*sigma_total - sigma_prev*sigma_prev)
                sigma_prev = sigma_cur
                filtered = cv2.GaussianBlur(
                    pyr_cur, ksize=(0, 0), sigmaX=sigma_cur)

            # both self.imgs and self.imgs_filtered contain filtered images in the scale space
            self.imgs.append(filtered)
            self.imgs_filtered.append(filtered)

            if i < self.num_levels-1:
                # resize the filtered frame
                pyr_down = cv2.resize(
                    filtered, (0, 0), fx=inv_scale, fy=inv_scale)
                pyr_cur = pyr_down