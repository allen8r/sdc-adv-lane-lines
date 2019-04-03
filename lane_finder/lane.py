import numpy as np

class LaneLines():
  """Object used to consolidate attributes associated with detected
  lane lines
  """

  def __init__(self):
    # Lane lines detected x and y positions
    self.leftx = None
    self.lefty = None
    self.rightx = None
    self.righty = None

    # Lane lines second order fit polynomials
    self.left_fit = None
    self.right_fit = None
    
    # Generated x, y values for plotting
    self.ploty = None
    self.left_fitx = None
    self.right_fitx = None

    # Window boxes (upper left and lower right coords) used to capture lane pixels
    self.sliding_windows = []

    # histogram frequency counts of pixel density of lane lines in the lower-half of the image
    self.histogram_freqs = None



class LaneCurvature():

  def __init__(self, lane_lines=LaneLines()):
    self.left_x = lane_lines.leftx
    self.left_y = lane_lines.lefty
    self.right_x = lane_lines.rightx
    self.right_y = lane_lines.righty
    self.y_eval = np.max(lane_lines.ploty)
    
    self.left_curvature = 0
    self.right_curvature = 0

    # Calculate lane curvatures
    self.lane_curvatures()


  def lane_curvatures(self):
    """
    TODO: document this
    """
    y_m_per_pixel = 30/720
    x_m_per_pixel = 3.7/700

    # Fit polynomials to real-world measurements
    left_fit_curve = np.polyfit(self.left_y*y_m_per_pixel, self.left_x*x_m_per_pixel, 2)
    right_fit_curve = np.polyfit(self.right_y*y_m_per_pixel, self.right_x*x_m_per_pixel, 2)

    # Calculate radii of curvature
    left_curve_rad = self.calc_curvature(left_fit_curve, self.y_eval, y_m_per_pixel)
    right_curve_rad = self.calc_curvature(right_fit_curve, self.y_eval, y_m_per_pixel)
    
    self.left_curvature = left_curve_rad
    self.right_curvature = right_curve_rad

    return self.left_curvature, self.right_curvature

  def calc_curvature(self, polyfit, d, conv_factor):
    """
    Calculates the lane curvature in the real-world measurement
    
    arguments:
    ployfit -- the lane line 2-degree polynomial curve function fit to the lane line pixels
    d -- the dimension value to be evaluated in the polyfit function
    conv_factor -- the conversion factor from pixels to real-world measurement (e.g. meters/pixel)
    
    returns the radius of curvature in the real-world measurement based on the provided conversion factor
    """
    return ((1 + (2*polyfit[0]*d*conv_factor + polyfit[1])**2)**1.5) / np.absolute(2*polyfit[0])