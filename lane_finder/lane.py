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

    # histogram of pixel density of lane lines in the lower-half of the image
    self.histogram = None
    
    