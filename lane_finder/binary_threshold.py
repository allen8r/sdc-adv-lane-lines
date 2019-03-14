import cv2
import numpy as np

class BinaryThresholder():
  
  def __init__(self):
    self.combined = None
    
    self.gray = None
    
    self.gradxy_binary = { 'x': None, 'y': None }
    
    self.mag_binary = None
    self.dir_binary = None
    
    self.hls_binary = { 'h_chan': None, 'l_chan': None, 's_chan': None }
    self.lab_binary = { 'l_chan': None, 'a_chan': None, 'b_chan': None }
    self.rgb_binary = { 'r_chan': None, 'g_chan': None, 'b_chan': None }
    
  def combine_thresholds(self, image,
                          sobel_kernel_size=5,
                          grad_thresh={'x': (20, 100), 'y': (20, 100)},
                          magnitude_thresh=(50, 70),
                          grad_dir_thresh=(0.65, 1.25), # in radians
                          hls_thresh={'h': ( 70, 150), 'l': (150, 255), 's': (180, 255)},
                          lab_thresh={'l': (150, 255), 'a': (100, 200), 'b': (100, 150)},
                          rgb_thresh={'r': (150, 255), 'g': (100, 200), 'b': ( 80, 200)}
                          ):

    self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # just for visual reference

    # Apply each of the thresholding functions
    self.gradxy(image, sobel_kernel_size, grad_thresh)

    self.mag_binary = self.mag_thresh(image, sobel_kernel=sobel_kernel_size, thresh=magnitude_thresh)

    self.dir_binary = self.dir_threshold(image, sobel_kernel=sobel_kernel_size, thresh=grad_dir_thresh)
    
    self.all_hls(image, hls_thresh)
    self.all_lab(image, lab_thresh)
    self.all_rgb(image, rgb_thresh)

    # Only using a select number of color and gradient filters; can play around with combining others
    self.combined = np.zeros_like(self.dir_binary)
    self.combined[((self.gradxy_binary['x'] == 1) & (self.gradxy_binary['y'] == 1)) | \
             ((self.mag_binary == 1) & (self.dir_binary == 1)) | \
             ((self.hls_binary['s'] == 1) & (self.hls_binary['h'] == 0)) | \
             ((self.lab_binary['b'] == 0)) | \
             ((self.rgb_binary['r'] == 1) & (self.rgb_binary['b'] == 0))] = 1
    
    return self.combined


  def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Calculate the directional gradient and apply the threshold"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8((255 * abs_sobel) / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

  def gradxy(self, image, sobel_kernel_size=3, grad_thresh=(0, 255)):
    self.gradxy_binary['x'] = self.abs_sobel_thresh(image, orient='x', 
                                                    sobel_kernel=sobel_kernel_size,
                                                    thresh=grad_thresh['x'])
    self.gradxy_binary['y'] = self.abs_sobel_thresh(image, orient='y',
                                                    sobel_kernel=sobel_kernel_size,
                                                    thresh=grad_thresh['y'])

  def mag_thresh(self, image, sobel_kernel=3, thresh=(0, 255)):
    '''Calculate the gradient magnitude and apply the threshold'''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled = np.uint8((255 * mag_sobelxy) / np.max(mag_sobelxy))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return mag_binary

  def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''Calculate gradient direction and apply the threshold'''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobelx = np.sqrt(np.square(sobelx))
    mag_sobely = np.sqrt(np.square(sobely))
    grad_dir = np.uint8(np.arctan2(mag_sobely, mag_sobelx))
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary

  def hls_threshold(self, image, channel='h', thresh=(90, 255)):
    """Calculate the color threshold for the requested channel
    in the HLS (or HSL) color space and apply the threshold
    
    arguments:
    image -- a color image in RGB format
    channel -- the requested channel to use for applying the threshold values
    thresh -- min and max (inclusive) values for the range of the threshold to apply
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    layer = H
    if channel == 'l' or channel == 'L':
        layer = L
    elif channel == 's' or channel == 'S':
        layer = S
        
    hls_binary = np.zeros_like(layer)
    hls_binary[(layer > thresh[0]) & (layer <= thresh[1])] = 1
    return hls_binary

  def all_hls(self, image, hls_thresh):
    self.hls_binary['h'] = self.hls_threshold(image, 'h', hls_thresh['h'])
    self.hls_binary['l'] = self.hls_threshold(image, 'l', hls_thresh['l'])
    self.hls_binary['s'] = self.hls_threshold(image, 's', hls_thresh['s'])

  def lab_threshold(self, image, channel='l', thresh=(90, 180)):
    '''Calculate the color threshold for the requested channel
    in the LAB color space and apply the threshold
    
    arguments:
    image -- a color image in RGB format
    channel -- the requested channel to use for applying the threshold values
    thresh -- min and max (inclusive) values for the range of the threshold to apply
    '''
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    L = lab[:,:,0]
    A = lab[:,:,1]
    B = lab[:,:,2]
    
    layer = L
    if channel == 'a' or channel == 'A':
        layer = A
    elif channel == 'b' or channel == 'B':
        layer = B
        
    lab_binary = np.zeros_like(layer)
    lab_binary[(layer > thresh[0]) & (layer <= thresh[1])] = 1
    return lab_binary

  def all_lab(self, image, lab_thresh):
    self.lab_binary['l'] = self.lab_threshold(image, 'l', lab_thresh['l'])
    self.lab_binary['a'] = self.lab_threshold(image, 'a', lab_thresh['a'])
    self.lab_binary['b'] = self.lab_threshold(image, 'b', lab_thresh['b'])

  def rgb_threshold(self, image, channel='r', thresh=(90, 255)):
    '''Calculate the color threshold for the requested channel
    in the RGB color space and apply the threshold
    
    arguments:
    image -- a color image in RGB format
    channel -- the requested channel to use for applying the threshold values
    thresh -- min and max (inclusive) values for the range of the threshold to apply
    '''
    rgb = image
    
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    
    layer = R
    if channel == 'g' or channel == 'G':
        layer = G
    elif channel == 'b' or channel == 'B':
        layer = B
        
    rgb_binary = np.zeros_like(layer)
    rgb_binary[(layer > thresh[0]) & (layer <= thresh[1])] = 1
    return rgb_binary
  
  def all_rgb(self, image, rgb_thresh):
    self.rgb_binary['r'] = self.rgb_threshold(image, 'r', rgb_thresh['r'])
    self.rgb_binary['g'] = self.rgb_threshold(image, 'r', rgb_thresh['g'])
    self.rgb_binary['b'] = self.rgb_threshold(image, 'r', rgb_thresh['b'])


