from src.datasets.generic import GroundTruth
from types import TracebackType
import numpy as np
import cv2
import math
import argparse
import io
from tqdm import tqdm 


from src.configuration import load_config, config_to_string, DEFAULTS as DEFAULT_CONFIGS

from src.vo.VisualOdometry import VisualOdometry

from src.camera import PinholeCamera
from src.datasets.archaeo import ArchaeoDataset, ArchaeoGroundTruth

# Features part
from src.features.types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from src.features.matcher import feature_matcher_factory, FeatureMatcherTypes

# VO part
from src.vo.FeaturesTracker import FeatureTrackerTypes, DescriptorFeatureTracker

# Visul
from src.plot import Mplot2d, Mplot3d
 
# from src.viewer import  Viewer3D
    
def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='pyVO')

    # Export directory,
    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='./scripts/base.ini')

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def config_to_string(config):
    with io.StringIO() as sio:
        config.write(sio)
        config_str = sio.getvalue()
    return config_str


def make_config(args):
    config = load_config(args.config, DEFAULT_CONFIGS["base"])
    
    print(config_to_string(config))
    
    return config


def make_dataloader(config):
    
    data_config = config["dataset"]

    # Data Loader
    print("Creating dataloaders for dataset in %s", data_config["name"])

    dataset = ArchaeoDataset(path=data_config.get("path"),
                             name=data_config.get("name"),
                             fps=data_config.getint("fps"),
                             type=data_config.get("type"))

    ground_truth = ArchaeoGroundTruth( path=data_config.get("ground_truth"),
                                       name=data_config.get("name"),
                                       type=data_config.get("type"))
    return dataset, ground_truth


def make_camera_model(config):
    
    camera_config = config["camera"]

    # Data Loader
    print("Creating Camera Model")

    camera = PinholeCamera( width=camera_config.getint("width"),
                            height=camera_config.getint("height"),
                            intrinsics=camera_config.getstruct("intrinsics"),
                            distortion_coeffs=camera_config.getstruct("distortion_coeffs"),
                            fps=camera_config.getint("fps")
                            )

    return camera


def main(args):

    # Load configuration
    config = make_config(args)

    # Load dataloader
    dataset, ground_truth = make_dataloader(config)

    camera = make_camera_model(config)

    # select your tracker configuration (see the file feature_tracker_configs.py)
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT

    # Init Tracker
        # Data Loader
    print("Init Tracker ")
    tracker_config = config["tracker"]
    feature_tracker = DescriptorFeatureTracker(num_features=tracker_config.getint("num_features"),
                                               num_levels=tracker_config.getint("num_levels"),
                                               scale_factor=tracker_config.getfloat("scale_factor"),
                                               
                                               detector_type=FeatureDetectorTypes.SIFT,
                                               descriptor_type=FeatureDescriptorTypes.SIFT,
                                               
                                               match_ratio_test=tracker_config.getfloat("match_ratio_test"),
                                               
                                               tracker_type=FeatureTrackerTypes.DES_BF
                                               )

    # create visual odometry object
    ground_truth = None
    vo = VisualOdometry(camera, ground_truth, feature_tracker)

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1

    # Draw 
    is_draw_3d = True
    plt3d = Mplot3d(title='3D trajectory')

    is_draw_err = True 
    err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')

    is_draw_matched_points = True 
    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    img_id = 0    
    
    for item in dataset:
        
        if item is None:
            continue
            
        # unpack
        img = item["img"]
        img_idx = item["idx"]

        if img is not None:
            
            # Track new frame
            vo.track(img, img_idx)  # main VO function
            
            if(img_id > 2):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                x_true, y_true, z_true = vo.traj3d_gt[-1]

                # draw 2D trajectory (on the plane xz)
                if is_draw_traj_img:
                    draw_x, draw_y = int(
                        draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    true_x, true_y = int(
                        draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
                    # estimated from green to blue
                    cv2.circle(traj_img, (draw_x, draw_y), 1,
                               (img_id*255/4540, 255-img_id*255/4540, 0), 1)
                    cv2.circle(traj_img, (true_x, true_y), 1,
                               (0, 0, 255), 1)  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    # show
                    cv2.imshow('Trajectory', traj_img)

                if is_draw_3d:           # draw 3d trajectory
                    plt3d.drawTraj(
                            vo.traj3d_gt, 'ground truth', color='r', marker='.')
                    plt3d.drawTraj(vo.traj3d_est, 'estimated',
                                       color='g', marker='.')
                    plt3d.refresh()

                if is_draw_err:         # draw error signals
                    errx = [img_id, math.fabs(x_true-x)]
                    erry = [img_id, math.fabs(y_true-y)]
                    errz = [img_id, math.fabs(z_true-z)]
                    err_plt.draw(errx, 'err_x', color='g')
                    err_plt.draw(erry, 'err_y', color='b')
                    err_plt.draw(errz, 'err_z', color='r')
                    err_plt.refresh()

                if is_draw_matched_points:
                    matched_kps_signal = [img_id, vo.num_matched_kps]
                    inliers_signal = [img_id, vo.num_inliers]
                    matched_points_plt.draw(
                        matched_kps_signal, '# matches', color='b')
                    matched_points_plt.draw(
                        inliers_signal, '# inliers', color='g')
                    matched_points_plt.refresh()

            # draw camera image
            cv2.imshow('Camera', vo.draw_img)

        # press 'q' to exit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_id += 1

    #print('press a key in order to exit...')
    # cv2.waitKey(0)

    if is_draw_traj_img:
        print('saving map.png')
        cv2.imwrite('map.png', traj_img)
    
    if is_draw_3d:
        plt3d.quit()

    if is_draw_err:
        err_plt.quit()
    
    if is_draw_matched_points is not None:
        matched_points_plt.quit()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = make_parser()

    main(parser.parse_args())
