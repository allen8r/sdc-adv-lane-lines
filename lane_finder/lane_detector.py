import numpy as np

from lane_finder.camera import Camera
from lane_finder.binary_threshold import BinaryThresholder
from lane_finder.persp_transform import PerspectiveTransformer
from lane_finder.lane import LaneLines
from lane_finder.lane import LaneCurvature
from lane_finder.visuals import Visualizer


class LaneDetector():

  def __init__(self):
    self.camera = Camera()
    self.camera.calibrate()

    self.binary_thresholder = BinaryThresholder()
    self.perspective_transformer = PerspectiveTransformer()
    self.visualizer = Visualizer()

    self.image = None
    self.corrected_image = None
    self.warp_corrected = None
    self.combined = None
    self.warped_combined = None

    self.lane_lines = LaneLines()
    self.lane_curvature = None
    self.lane_curve_rads = []
    self.vehichle_pos = None

    # Visuals
    self.result = None

    
    


  #############################################################################
  #
  # Lane Detection Pipeline
  #
  #############################################################################
  def detect_lane(self, image):
    # 1. Calibrate camera
    if not self.camera.CAM_CALIBRATED:
      self.camera.calibrate()

    self.image = image

    # 2. Correct camera distortions
    self.corrected_image = self.camera.correct_distortion(image)

    # 3. Combine gradient and color binary thresholds
    self.combined = self.binary_thresholder.combine_thresholds(self.corrected_image)

    # 4. Warp combined binary thresholds image to "bird's eye view"
    # first store warped corrected image for reference
    self.warped_corrected = self.perspective_transformer.warp_img(self.corrected_image)
    self.warped_combined = self.perspective_transformer.warp_img(self.combined)

    # 5. Detect lane lines using window boxing
    self.detect_lane_lines(self.warped_combined)

    # 6. Determine lane curvature
    self.lane_curvature = LaneCurvature(self.lane_lines)
    self.lane_curve_rads.append(self.lane_curvature.left_curvature)
    self.lane_curve_rads.append(self.lane_curvature.right_curvature)
    
    # 7. Determine vehicle position within lane
    self.vehichle_pos = self.position_offset()

    # 8. Mark and visualize lane with HUD info
    self.result = self.visualizer.draw_lane(
                              self.warped_combined,
                              self.corrected_image,
                              self.perspective_transformer.perspective_Minv,
                              self.lane_lines)

    self.result = self.visualizer.add_HUD_info(
                              self.result,
                              self.lane_curve_rads,
                              self.vehichle_pos)

    # 9. Clean up; reset thresholder once image is fully processed; 
    # get it ready for next image to be processed
    #self.reset_pipeline()

    return self.result



  


  def detect_lane_lines(self, warped_binary):
    # Take a histogram of the bottom half of the image
    histogram = self.get_histogram(warped_binary)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to hold left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    sliding_windows = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_binary.shape[0] - (window+1)*window_height
        win_y_high = warped_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Save window corners for drawing the windows on the visualization image
        sliding_windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
        sliding_windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    self.lane_lines.leftx = leftx
    self.lane_lines.lefty = lefty
    self.lane_lines.rightx = rightx
    self.lane_lines.righty = righty

    self.lane_lines.left_fit = left_fit
    self.lane_lines.right_fit = right_fit

    self.lane_lines.ploty = ploty
    self.lane_lines.left_fitx = left_fitx
    self.lane_lines.right_fitx = right_fitx
    
    self.lane_lines.sliding_windows = sliding_windows
    self.lane_lines.histogram = histogram

    return self.lane_lines


  def position_offset(self):
    '''Assume the camera is mounted at the center of the car,
    such that the lane center is the midpoint at the bottom of
    the image between the two lines detected. The offset of the
    lane center from the center of the image (converted from
    pixels to meters) is the distance from the center of the lane.
    '''
    x_m_per_pixel = 3.7/700

    y_eval = np.max(self.lane_lines.ploty)
    left = self.eval_poly2_fit(self.lane_lines.left_fit, y_eval)
    right = self.eval_poly2_fit(self.lane_lines.right_fit, y_eval)
    lane_center = (left + right) / 2
    car_center = self.image.shape[1] / 2
    
    offset = car_center - lane_center
    
    return offset * x_m_per_pixel


  def eval_poly2_fit(self, polyfit2, d):
    """
    ployfit2 -- the lane line 2-degree polynomial curve function fit to the lane line pixels
    d -- the dimension value to be evaluated in the polyfit function
    """
    return polyfit2[0]*d**2 + polyfit2[1]*d + polyfit2[2]

  def get_histogram(self, image):
    '''Retrieve histogram of pixel density for lane lines in lower half of the image'''
    return np.sum(image[image.shape[0]//2:,:], axis=0)

  def reset_pipeline(self):
    pass