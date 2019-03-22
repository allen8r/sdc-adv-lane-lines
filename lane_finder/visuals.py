import cv2
import numpy as np

class Visualizer():

  def __init__(self):
    self.visualized = None
    self.combined_binary = None
    self.warped_combined = None
    self.detected_lane_lines_warped = None
    self.lane_highlighted = None
    self.lane_highlighted_HUD_info = None
    

    self.RED = (255, 0, 0)
    self.GREEN = (0,255,0)
    self.BLUE = (0, 0, 255)
    self.YELLOW = (255, 255, 0)
    self.WHITE = (255,255,255)
    self.CYAN = (87, 220, 217)

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
    out_img[lane_lines.lefty, lane_lines.leftx] = self.RED
    out_img[lane_lines.righty, lane_lines.rightx] = self.BLUE

    #Draw sliding windows
    for win_corners in lane_lines.sliding_windows:
        l_x, b_y = win_corners[0]
        r_x, t_y = win_corners[1]
        
        # cv2 needs top-left and bottom-right points of the rectangle
        pt1 = (l_x, t_y)
        pt2 = (r_x, b_y)
        cv2.rectangle(out_img, pt1, pt2, self.GREEN, thickness=3)
        

    #Draw 2nd-degree polynomial fit curves
    curve_thickness = 6
    left_pts = np.dstack((lane_lines.left_fitx, lane_lines.ploty))
    left_pts = np.array(left_pts, np.int32)
    left_pts = left_pts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [left_pts], isClosed=False, color=self.YELLOW, thickness=curve_thickness)

    right_pts = np.dstack((lane_lines.right_fitx, lane_lines.ploty))
    right_pts = np.array(right_pts, np.int32)
    right_pts = right_pts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [right_pts], isClosed=False, color=self.YELLOW, thickness=curve_thickness)

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
    cv2.fillPoly(color_warp, np.int_([pts]), self.GREEN)

    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, persp_Minv, (corrected.shape[1], corrected.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(corrected, 1, newwarp, 0.3, 0)
    
    self.lane_highlighted = result
    
    return self.lane_highlighted

  def add_HUD_info(self, image, lane_curve_rads, vehicle_position, histogram):
    overlay = np.zeros_like(image).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontsize_mult = 1
    fontthickness = 2
    linetype = cv2.LINE_AA
    
    # Add curvature data
    l_curvature = "{:>5.3f} km".format(lane_curve_rads[0]/1000)
    r_curvature = "{:>5.3f} km".format(lane_curve_rads[1]/1000)
    
    overlay_width = overlay.shape[1]
    overlay_height = overlay.shape[0]

    left_align = overlay_width - 400
    col1 = left_align + 50
    col2 = col1 + 75

    top_align = 50
    row_height = 40
    row1 = top_align
    row2 = row1 + row_height
    row3 = row2 + row_height
    
    cv2.putText(overlay, 'Curvature Radius:', (col1, row1), font, fontsize_mult, self.CYAN, fontthickness, linetype)
    cv2.putText(overlay, '  L', (col1, row2), font, fontsize_mult, self.CYAN, fontthickness, linetype)
    cv2.putText(overlay, l_curvature, (col2, row2), font, fontsize_mult, self.CYAN, fontthickness, linetype)
    cv2.putText(overlay, '  R', (col1, row3), font, fontsize_mult, self.CYAN, fontthickness, linetype)
    cv2.putText(overlay, r_curvature, (col2, row3), font, fontsize_mult, self.CYAN, fontthickness, linetype)
    
    # Add vehicle position data (offset from center of lane)
    middle = overlay_width // 2
    arrow_x = middle - 70
    arrow = '<--'
    position = "{:>+5.2f} m".format(vehicle_position)
    
    if vehicle_position > 0:
        arrow = '-->'
        arrow_x = middle
        
    veh_pos_y = 600
    
    cv2.putText(overlay, position, (middle-70, veh_pos_y), font, fontsize_mult, self.WHITE, fontthickness, linetype)
    cv2.putText(overlay, '|', (middle, veh_pos_y+row_height), font, fontsize_mult, self.WHITE, fontthickness, linetype)
    cv2.putText(overlay, arrow, (arrow_x, veh_pos_y+row_height), font, fontsize_mult, self.WHITE, fontthickness, linetype)
    
    # Add binary image visuals
    scaled_combined_binary = self.scale(self.increase_channels(self.combined_binary))
    scaled_warped_combined = self.scale(self.increase_channels(self.warped_combined))
    scaled_sliding_windows = self.scale(self.detected_lane_lines_warped)

    img_width = image.shape[1]
    img_height = image.shape[0]
    scaled_histogram = np.zeros((img_height, img_width, 3))
    plotx = np.linspace(0, img_width-1, img_width)

    curve_thickness = 6
    pts = np.dstack((plotx, (img_height - 200) - histogram)) # invert the plotting
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(scaled_histogram, [pts], isClosed=False, color=self.CYAN, thickness=curve_thickness)
    scaled_histogram = self.scale(scaled_histogram)

    scaled_width = scaled_combined_binary.shape[1]
    scaled_height = scaled_combined_binary.shape[0]
    binary_strip = np.zeros((scaled_height*4, scaled_width, 3))
    binary_strip[0:scaled_height,0:scaled_width,:] = scaled_combined_binary
    binary_strip[scaled_height:2*scaled_height,0:scaled_width,:] = scaled_warped_combined
    binary_strip[2*scaled_height:3*scaled_height,0:scaled_width,:] = scaled_sliding_windows
    binary_strip[3*scaled_height:4*scaled_height,0:scaled_width,:] = scaled_histogram


    self.lane_highlighted_HUD_info = cv2.addWeighted(image, 1, overlay, 1, 0)
    self.lane_highlighted_HUD_info[0:binary_strip.shape[0],0:binary_strip.shape[1],:] = binary_strip
    
    return self.lane_highlighted_HUD_info


  def scale(self, image, factor=0.2):
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

  def increase_channels(self, binary_image):
    return np.dstack((binary_image, binary_image, binary_image)) * 255


  def save_image(self, image, filename):
    print(f"Saving {filename}")
    # NOTE: image[:, :, ::-1] reverses the (default BRG channels
    # from cv2.imwrite) values to RGB (what we expect to see)
    cv2.imwrite(f"{filename}", image[:, :, ::-1])