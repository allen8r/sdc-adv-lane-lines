## Writeup Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist_checker]: ./output_images/undistorted-checkerboard.png "Undistorted Checkerboard"
[undistorted_road]: ./output_images/undistorted-road.png "Undistorted road sample"
[road_normal]: ./output_images/corrected-bounded.png "Road Normal"
[road_transformed]: ./output_images/warped-bounded.png "Road Transformed"
[thresholding_binary]: ./output_images/thresholding.png "Thresholding Binary"
[fit_curves]: ./output_images/sliding-window-polyfit.png "Fit Visual"
[histogram]: ./output_images/histogram.png "Histogram"
[marked_lane]: ./output_images/lane_marked.png "Sample Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / READMES

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the Jupyter notebook `./find-lane-lines.ipynb` in the function `calibrate_camera()`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_pt` is just a replicated array of coordinates, and `obj_pts` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_pts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_pts` and `img_pts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result (See `correct_distortion()` function in code cell #3 in the notebook): 

![alt text][undist_checker]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I apply the distortion correction to one of the test images like this one:
![alt text][undistorted_road]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in the `combine_threshold()` function in code cell# 10. Here's an example of my output for this step.

![alt text][thresholding_binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_img()`, which appears in code cell# 11 in the file `find-lane-lines.ipynb`. The `warp_img()` function takes as inputs an image (`image`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcoded the source and destination points in the following manner:

```python
    stlx =  585
    strx =  700
    st_y =  455
    sblx =  200
    sbrx = 1110
    sb_y =  720

    src = np.float32([[strx, st_y],[sbrx, sb_y],
                      [sblx, sb_y],[stlx, st_y]])

    dtlx =  320
    dtrx =  960
    dt_y =    0
    dblx = dtlx
    dbrx = dtrx
    db_y =  720

    dst = np.float32([[dtrx, dt_y],[dbrx, db_y],
                      [dblx, db_y],[dtlx, dt_y]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:--------------|:--------------| 
| 700, 455      | 960, 0        | 
| 1110, 720     | 960, 720      |
| 200, 720      | 0, 720        |
| 585, 455      | 320, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

|                               |                               
|:------------------------------|
|*Test Image*                   |
|![alt text][road_normal]       |
|                               |
|*Warped Counterpart*           |
|![alt_text][road_transformed]  |
|                               |

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To fit my lane lines with a 2nd order polynomial I used the sliding-window technique on the warped binary image in the `detect_lane_line()` function in code cell# 16. The histogram of the pixel counts along the x-axis provides starting points for the sliding window search for lane pixels where the highest pixel counts indicate a high likelihood of the existence of lane lines on the road.

|                                               |
|:----------------------------------------------|
|*2nd order polynomial fit curves*              |
|![alt text][fit_curves]                        |
|                                               |
|*Histogram of pixel counts along the x-axis*   |
|![alt text][histogram]                         |
|                                               |

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Lane curavature was calculated in code cell# 19 of the notebook in the lane_curavatures() function. The function takes in (x,y) locations of identified lane pixels, performs 2nd order polynomial fits on the left and right lane pixels, and evaluates the curvature values at the max y value for the curves.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`draw_lane()` function in code cell# 20 of the `find-lane-lines.ipynb` plots the fit curves back onto the road, identifying the lane area.  Here is an example of my result on a test image:

![alt text][marked_lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_output.mp4)

Here's a [link to my challenge video result](./output_videos/challenge_output.mp4) The challenge video can use some work as demonstrated by the very visible twisting and winding of the green lane marking.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One major issue I faced working on this project was trying to use openCV to create some of the graphics. One example was trying to make visualizations of the sliding windows by plotting the green boxes onto the stacked image created for the output result. I just could not get the image to display the green window boxes nor to show the red and blue pixels for the lane line pixels detected. Yet, all the data was there and fit curves were generated properly. In order to move on, I had to abandon using openCV for the plotting of these visuals and ended up turning to matplotlib for the displays. This inability to plot certain things with openCV wasted a lot of my time; but I'm pretty sure this problem was due to pilot error on my part rather than the library having a bug. I'll need to revisit the code to see what I was doing wrong in the stacked layer image. Fortunately, on the color images from the video, openCV plotting APIs was doing their job as expected.

The pipeline is most likely to fail in the lane pixel detection area. The lane detection is only as good as there are pixels available for detection. This pixel density available for detection is highly dependent of the binary pixel images. So, a likely area for improvement is to tune the color and gradient thresholding step to better expose the lane pixels. This area was, in fact, what was causing my pipeline to fail initially, markedly around the 20-25s time window of the input video. Here, the lighter pavement color, along with the bright sunlight, seem to blend in and hide the yellow lane line on the left. The lane detection was mistaking the line formed by the median barrier as the left lane line. To fix this, I extracted the frame from the video that was failing and used it as my test image to iteratively fine tune the various thresholding functions until the pipeline was able to distinguish the yellow left lane line.

For future improvements, adding in an alternative, quicker search for lane pixels once a high-confidence detection is established by the sliding-window detection would be desirable. Using tracking for detected lane lines in past frames and doing some sort of averaging would help ensure a smoother lane marking. In the coding side, creating some class objects to consolidate some of the data would help make the code a lot cleaner and remove some redundant function parameters/arguments.
