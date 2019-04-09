import cv2
import numpy as np


class Visualizer():

  def __init__(self):
    self.v_elems = VisualElements()

    self.visualized = None
    self.combined_binary = None
    self.warped_combined = None
    self.detected_lane_lines_warped = None
    self.lane_highlighted = None
    self.lane_highlighted_HUD_info = None
    self.lane_lines_histogram = None
    
  def draw_polygon(self, image, vertices, color=(255,0,0)):
    """Draw polygon formed by the provide vertices onto the provided image"""

    draw_canvas = np.zeros_like(image).astype(np.uint8)
    
    if image.ndim < 3: # basically if it's only 2d so is a grayscale image
      canvas_zeros = np.zeros_like(image).astype(np.uint8)
      draw_canvas = np.dstack((canvas_zeros, canvas_zeros, canvas_zeros))
      
      # Convert image to RGB to allow adding in color lines for the polygon
      image = np.dstack((image, image, image)) * 255
      #image = image * 255
      #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Draw the polygon onto the blank canvas
    pts = vertices
    pts = np.array(vertices, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(draw_canvas, [pts], isClosed=True, color=color, thickness=6)

    # overlay the polygon canvas onto the original image
    self.visualized = cv2.addWeighted(image, 1, draw_canvas, 1, 0)

    return self.visualized


  def visualize_detected_lane_lines(self, warped_binary, lane_lines):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_binary, warped_binary, warped_binary)) * 255

    #Draw lane line pixels
    out_img[lane_lines.lefty, lane_lines.leftx] = self.v_elems.LITE_BLUE
    out_img[lane_lines.righty, lane_lines.rightx] = self.v_elems.RED

    #Draw sliding windows
    for win_corners in lane_lines.sliding_windows:
        l_x, b_y = win_corners[0]
        r_x, t_y = win_corners[1]
        
        # cv2 needs top-left and bottom-right points of the rectangle
        pt1 = (l_x, t_y)
        pt2 = (r_x, b_y)
        cv2.rectangle(out_img, pt1, pt2, self.v_elems.GREEN, thickness=4)
        
    #Draw 2nd-degree polynomial fit curves
    curve_thickness = 9
    left_pts = np.dstack((lane_lines.left_fitx, lane_lines.ploty))
    left_pts = np.array(left_pts, np.int32)
    left_pts = left_pts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [left_pts], isClosed=False, color=self.v_elems.YELLOW, thickness=curve_thickness)

    right_pts = np.dstack((lane_lines.right_fitx, lane_lines.ploty))
    right_pts = np.array(right_pts, np.int32)
    right_pts = right_pts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [right_pts], isClosed=False, color=self.v_elems.YELLOW, thickness=curve_thickness)

    self.detected_lane_lines_warped = out_img
    
    return self.detected_lane_lines_warped


  def draw_lane(self, warped_img, corrected, persp_Minv, lane_lines):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lane_lines.left_fitx, lane_lines.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lane_lines.right_fitx, lane_lines.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), self.v_elems.GREEN)

    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, persp_Minv, (corrected.shape[1], corrected.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(corrected, 1, newwarp, 0.3, 0)
    
    self.lane_highlighted = result
    
    return self.lane_highlighted


  def add_HUD_info(self, image, lane_curve_rads, vehicle_position, histogram_freqs):
    # Add vehicle position data (offset from center of lane)
    overlay = np.zeros_like(image).astype(np.uint8)
    overlay_width = overlay.shape[1]
    overlay_height = overlay.shape[0]

    middle = overlay_width // 2
    arrow_x = middle - 55
    arrow = '<--'
    position = "{:>+5.2f} m".format(vehicle_position)
    
    if vehicle_position > 0:
        arrow = '-->'
        arrow_x = middle
  
    veh_pos_y = 620
    
    row_height = self.v_elems.row_height
    font = self.v_elems.font
    fontsize_mult = self.v_elems.fontsize_mult
    color = self.v_elems.WHITE
    fontthickness = self.v_elems.fontthickness
    linetype = self.v_elems.linetype

    cv2.putText(overlay, position, (middle-55, veh_pos_y), font, fontsize_mult, color, fontthickness, linetype)
    cv2.putText(overlay, '|', (middle, veh_pos_y+row_height), font, fontsize_mult, color, fontthickness, linetype)
    cv2.putText(overlay, arrow, (arrow_x, veh_pos_y+row_height), font, fontsize_mult, color, fontthickness, linetype)
    
    self.lane_highlighted_HUD_info = cv2.addWeighted(image, 1, overlay, 1, 0)

    details_panel = self.create_details_panel(image, lane_curve_rads, histogram_freqs)
    self.lane_highlighted_HUD_info[0:details_panel.shape[0],0:details_panel.shape[1],:] = details_panel
    
    return self.lane_highlighted_HUD_info


  def create_details_panel(self, image, lane_curve_rads, histogram_freqs):
    # Add binary image visuals
    scaled_combined_binary = self.scale(self.decorate_display(self.combined_binary, "BINARY THRESHOLDS", True))
    scaled_warped_combined = self.scale(self.decorate_display(self.warped_combined, "WARPED BINARY", True))
    scaled_sliding_windows = self.scale(self.decorate_display(self.detected_lane_lines_warped, "SLIDING WINDOWS"))

    self.lane_lines_histogram = self.plot_histogram(image, histogram_freqs)
    scaled_histogram = self.scale(self.decorate_display(self.lane_lines_histogram, "LANE LINE PIXELS"))

    scaled_width = scaled_combined_binary.shape[1]
    scaled_height = scaled_combined_binary.shape[0]
    
    details_panel_bg = image[:image.shape[0], :scaled_width, :]
    details_panel = np.zeros((image.shape[0], scaled_width, 3), dtype=np.uint8)

    details_panel[0:scaled_height,0:scaled_width,:] = scaled_combined_binary
    details_panel[scaled_height:2*scaled_height,0:scaled_width,:] = scaled_warped_combined
    details_panel[2*scaled_height:3*scaled_height,0:scaled_width,:] = scaled_sliding_windows
    details_panel[3*scaled_height:int(3*scaled_height+scaled_histogram.shape[0]),0:scaled_width,:] = scaled_histogram

    # Add curvature data
    l_curvature = f"{lane_curve_rads[0]/1000:5.2f}"
    l_curvature = f"{l_curvature:>6}"
    r_curvature = f"{lane_curve_rads[1]/1000:5.2f}"
    r_curvature = f"{r_curvature:>6}"
    
    left_align = 0
    col1 = left_align + 10
    col2 = col1 + 75

    top_align = 620
    row_height = self.v_elems.row_height
    row1 = top_align
    row2 = row1 + row_height
    row3 = row2 + row_height
    
    font = self.v_elems.font
    fontsize_mult = self.v_elems.fontsize_mult
    color = self.v_elems.CYAN
    fontthickness = self.v_elems.fontthickness
    linetype = self.v_elems.linetype

    cv2.putText(details_panel, 'Curv. Rad. (km):', (col1, row1), font, fontsize_mult, color, fontthickness, linetype)
    cv2.putText(details_panel, '  L', (col1, row2), font, fontsize_mult,  color, fontthickness, linetype)
    cv2.putText(details_panel, l_curvature, (col2, row2), font, fontsize_mult,  color, fontthickness, linetype)
    cv2.putText(details_panel, '  R', (col1, row3), font, fontsize_mult,  color, fontthickness, linetype)
    cv2.putText(details_panel, r_curvature, (col2, row3), font, fontsize_mult,  color, fontthickness, linetype)

    # Combine with the background to create a tinted panel effect
    details_panel = cv2.addWeighted(details_panel, 0.9, details_panel_bg, 0.10, 0)

    return details_panel


  def plot_histogram(self, image, histogram_freqs):
    img_width = image.shape[1]
    img_height = image.shape[0]
    padding = 175
    #img_height = int(image.shape[0] / 2 + padding)
    histogram = np.zeros((img_height, img_width, 3))
    plotx = np.linspace(0, img_width-1, img_width)

    pts = np.dstack((plotx, (img_height - padding) - histogram_freqs)) # invert the plotting
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(histogram, [pts], isClosed=False, color=self.v_elems.GREEN, thickness=4)
    
    return histogram


  def scale(self, image, factor=0.2):
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


  def increase_channels(self, binary_image):
    """Returns a 3-channel image based on the provided binary image.
    Assumes the provided image is a single-channel image.
    """
    return np.dstack((binary_image, binary_image, binary_image)) * 255


  def draw_border(self, image, color=(87, 220, 217)):
    decorated = image
    lower_right_corner = (decorated.shape[1], decorated.shape[0])

    # draw black edge border
    cv2.rectangle(decorated, (0, 0), lower_right_corner, (0, 0, 0), thickness=30)

    # draw highlight border
    offset = 20
    lrc_x = lower_right_corner[0] - offset
    lrc_y = lower_right_corner[1] - offset
    cv2.rectangle(decorated, (offset, offset), (lrc_x, lrc_y), color, thickness=8)

    return decorated


  def label_display(self, image, label):
    decorated = image
    position = (35, image.shape[0]-35)

    cv2.putText(decorated, label, position, 
                self.v_elems.font, 1.75,
                self.v_elems.CYAN, self.v_elems.fontthickness,
                self.v_elems.linetype)

    return decorated


  def decorate_display(self, image, label, binary_img=False):
    decorated = image
    if binary_img:
      decorated = self.increase_channels(decorated)
    decorated = self.label_display(self.draw_border(decorated), label)
    return decorated


  def save_image(self, image, filename):
    print(f"Saving {filename}")
    # NOTE: image[:, :, ::-1] reverses the (default BRG channels
    # from cv2.imwrite) values to RGB (what we expect to see)
    cv2.imwrite(f"{filename}", image[:, :, ::-1])


class VisualElements():
  def __init__(self):
    self.RED = (255, 0, 0)
    self.GREEN = (0,255,0)
    self.BLUE = (0, 0, 255)
    self.LITE_BLUE = (0, 153, 255)
    self.YELLOW = (255, 255, 0)
    self.WHITE = (255,255,255)
    self.CYAN = (87, 220, 217)
    self.PURPLE = (153, 0, 153)

    self.font = cv2.FONT_HERSHEY_SIMPLEX
    self.fontsize_mult = 0.80
    self.fontthickness = 2
    self.linetype = cv2.LINE_AA
    self.italics = cv2.FONT_ITALIC

    self.row_height = 40

