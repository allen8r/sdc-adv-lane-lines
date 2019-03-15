import cv2
import numpy as np

class Visualizer():

  def __init__(self):
    self.visualized = None

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

  def visualize_detected_lanes(self, warped_binary, lane_lines):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_binary, warped_binary, warped_binary)) * 255

    #Draw lane line pixels
    red = [255, 0, 0]
    blue = [0, 0, 255]
    out_img[lane_lines.lefty, lane_lines.leftx] = red
    out_img[lane_lines.righty, lane_lines.rightx] = blue

    #Draw sliding windows
    green = [0, 255, 0]
    for win_corners in lane_lines.sliding_windows:
        l_x, b_y = win_corners[0]
        r_x, t_y = win_corners[1]
        
        # cv2 needs top-left and bottom-right points of the rectangle
        pt1 = (l_x, t_y)
        pt2 = (r_x, b_y)
        cv2.rectangle(out_img, pt1, pt2, green, thickness=3)
        

    #Draw 2nd-degree polynomial fit curves
    yellow = [255, 255, 0]
    curve_thickness = 7
    
    left_pts = np.dstack((lane_lines.left_fitx, lane_lines.ploty))
    left_pts = np.array(left_pts, np.int32)
    left_pts = left_pts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [left_pts], isClosed=False, color=yellow, thickness=curve_thickness)

    right_pts = np.dstack((lane_lines.right_fitx, lane_lines.ploty))
    right_pts = np.array(right_pts, np.int32)
    right_pts = right_pts.reshape((-1, 1, 2))
    cv2.polylines(out_img, [right_pts], isClosed=False, color=yellow, thickness=curve_thickness)

    self.visualized = out_img
    
    return self.visualized
    


  def save_image(self, filename):
    print(f"Saving {filename}")
    # NOTE: self.visualized[:, :, ::-1] reverses the (default BRG channels
    # from cv2.imwrite) values to RGB (what we expect to see)
    img = self.visualized[:, :, ::-1]
    cv2.imwrite(f"{filename}", img)