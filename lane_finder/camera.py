import cv2
import glob
import matplotlib.image as mpimg
import numpy as np

class Camera():
  def __init__(self):
      self.camera_matrix = None
      self.distortion_coefficients = None
      self.CAM_CALIBRATED = False
    
  def calibrate(self, img_dir='camera_cal', nx=9, ny=6):
      """
      img_dir - directory name where the list of calibration images for the camera reside
      nx       - number of 'checkerboard' squares in the horizontal
      ny       - number of 'checkerboard' squares in the vertical
      """
      if not self.CAM_CALIBRATED:
        imgs = glob.glob(f'./{img_dir}/calibration*.jpg')

        # Collect object points and image points for use in calibrating camera
        obj_pts = [] # 3D points in the real world space
        img_pts = [] # 2D points in the image plane
        sample_img = None

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... , (8,5,0)
        obj_pt = np.zeros((ny * nx, 3), np.float32)
        # z-coordinate will stay zero; but x, y coordinates need to be generated
        obj_pt[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        for fname in imgs:
          img = mpimg.imread(fname)

          if sample_img is None:
            sample_img = img

          gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

          found, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

          # If corners are found, add object points and image points
          if found:            
            img_pts.append(corners)
            obj_pts.append(obj_pt)
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, found)

        ret, cam_mtx, dist_coeffs, rot_vecs, trans_vecs = \
            cv2.calibrateCamera(obj_pts, img_pts, sample_img.shape[1::-1], None, None)

        self.camera_matrix = cam_mtx
        self.distortion_coefficients = dist_coeffs
        self.CAM_CALIBRATED = True

      
  def correct_distortion(self, image):
      """
      Corrects for camera distortions in the provided image based on the camera matrix 
      and distortion coefficients from the calibrated camera.
      
      Returns the corrected, undistorted image.
      """ 
      return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, None)