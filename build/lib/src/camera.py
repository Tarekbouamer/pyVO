import numpy as np 
import cv2
#import g2o
from src.utils.geometric import add_ones


class Camera: 
    def __init__(self, width, height, intrinsics, distortion_coeffs, fps): 
        
        self.width = width
        self.height = height
        
        # intrinsics = [fx, fy, cx, cy]
        self.fx, self.fy, self.cx, self.cy = intrinsics
        
        # distortion_coeffs = [k1, k2, p1, p2, k3]
        self.D = np.array(distortion_coeffs,dtype=np.float32)   
        
        if fps:
            self.fps = fps 
        else:
            self.fps =1 
        
        self.is_distorted = np.linalg.norm(self.D) > 1e-10
        self.initialized = False     
        
        
class PinholeCamera(Camera):
    def __init__(self, width, height, intrinsics, distortion_coeffs, fps):
        super().__init__(width, height, intrinsics, distortion_coeffs, fps)

        # intrinsics = [fx, fy, cx, cy]        
        fx, fy, cx, cy = intrinsics

        self.K = np.array([[fx,        0,      cx ],
                           [ 0,        fy,     cy ],
                           [ 0,        0,      1  ]])
        
        self.Kinv = np.array([[ 1/fx,    0,     -cx/fx],
                              [   0,    1/fy,   -cy/fy],
                              [   0,     0,        1  ]])             
        
        self.u_min, self.u_max = 0, width
        self.v_min, self.v_max = 0, height       
        self.init()    
        
    def init(self):
        if not self.initialized:
            self.initialized = True 
            self.undistort_image_bounds()        
                
    # project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # out: Nx2 image points, [Nx1] array of map point depths     
    def project(self, xcs):
        #u = self.fx * xc[0]/xc[2] + self.cx
        #v = self.fy * xc[1]/xc[2] + self.cy  
        projs = self.K @ xcs.T     
        zs = projs[-1]      
        projs = projs[:2]/ zs   
        return projs.T, zs
        
    # unproject 2D point uv (pixels on image plane) on 
    def unproject(self, uv):
        x = (uv[0] - self.cx)/self.fx
        y = (uv[1] - self.cy)/self.fy
        return x,y

    # in:  uvs [Nx2]
    # out: xcs array [Nx3] of normalized coordinates     
    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]        

    # in:  uvs [Nx2]
    # out: uvs_undistorted array [Nx2] of undistorted coordinates  
    def undistort_points(self, uvs):
        if self.is_distorted:
            #uvs_undistorted = cv2.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)   # =>  Error: while undistorting the points error: (-215:Assertion failed) src.isContinuous() 
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)            
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs 
        
    # update image bounds     
    def undistort_image_bounds(self):
        uv_bounds = np.array([[self.u_min, self.v_min],
                                [self.u_min, self.v_max],
                                [self.u_max, self.v_min],
                                [self.u_max, self.v_max]], dtype=np.float32).reshape(4,2)
        if self.is_distorted:
                uv_bounds_undistorted = cv2.undistortPoints(np.expand_dims(uv_bounds, axis=1), self.K, self.D, None, self.K)      
                uv_bounds_undistorted = uv_bounds_undistorted.ravel().reshape(uv_bounds_undistorted.shape[0], 2)
        else:
            uv_bounds_undistorted = uv_bounds 
        self.u_min = min(uv_bounds_undistorted[0][0],uv_bounds_undistorted[1][0])
        self.u_max = max(uv_bounds_undistorted[2][0],uv_bounds_undistorted[3][0])        
        self.v_min = min(uv_bounds_undistorted[0][1],uv_bounds_undistorted[2][1])    
        self.v_max = max(uv_bounds_undistorted[1][1],uv_bounds_undistorted[3][1])  
    
     
    def is_in_image(self, uv, z):
        return (uv[0] > self.u_min) & (uv[0] < self.u_max) & \
               (uv[1] > self.v_min) & (uv[1] < self.v_max) & \
               (z > 0)         
                
    # input: [Nx2] array of uvs, [Nx1] of zs 
    # output: [Nx1] array of visibility flags             
    def are_in_image(self, uvs, zs):
        return (uvs[:, 0] > self.u_min) & (uvs[:, 0] < self.u_max) & \
               (uvs[:, 1] > self.v_min) & (uvs[:, 1] < self.v_max) & \
               (zs > 0 )