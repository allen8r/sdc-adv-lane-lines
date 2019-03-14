import cv2
import numpy as np

class PerspectiveTransformer():

  # Source image vertices for area of interest
  # s(rc)t(op)l(eft)x
  stlx =  585
  # s(rc)t(op)r(ight)x
  strx =  700
  # s(rc)t(op)y
  st_y =  455
  # s(rc)b(ottom)l(eft)x
  sblx =  200
  # s(rc)b(ottom)r(ight)x
  sbrx = 1110
  # s(rc)b(ottom)y
  sb_y =  720
  src_default = np.float32([[strx, st_y],[sbrx, sb_y],
                    [sblx, sb_y],[stlx, st_y]])

  # Destination image vertices for transformed area of interest  
  # d(est)t(op)l(eft)x
  dtlx =  320
  # d(est)t(op)r(ight)x
  dtrx =  960
  # d(est)t(op)y
  dt_y =    0
  # d(est)b(ottom)l(eft)x
  dblx = dtlx
  # d(est)b(ottom)r(ight)x
  dbrx = dtrx
  # d(est)b(ottom)y
  db_y =  720
  dst_default = np.float32([[dtrx, dt_y],[dbrx, db_y],
                    [dblx, db_y],[dtlx, dt_y]])

  def __init__(self):
    self.warped = None
    self.perspective_M = None # M matrix
    self.perspective_Minv = None # M inverse matrix
    self.src = None
    self.dst = None
    

  def warp_img(self, image, src=src_default, dst=dst_default):
    """
    Warps the provided image based on the provided src and dst points.
    """
    self.src = src
    self.dst = dst

    self.perspective_M = cv2.getPerspectiveTransform(src, dst)
    self.perspective_Minv = cv2.getPerspectiveTransform(dst, src) # inverse matrix for reversing transforms
    img_size = (image.shape[1], image.shape[0])
    self.warped = cv2.warpPerspective(image, self.perspective_M, img_size, flags=cv2.INTER_LINEAR)
    
    return self.warped
    

