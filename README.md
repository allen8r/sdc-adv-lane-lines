# Advanced Lane Finding

![Detected Lane](./output_videos/refactored_output.gif)

The goal of this project is to write a software pipeline to identify the lane boundaries in a traffic video from the perspective of a moving vehicle from which the video was captured. A [writeup](./writeup_report.md) of the project describes the details of the lane detection pipeline.

Note that the writeup describes the work done on the original prototype version of the project. All orginal work was done in a single lengthy Jupyter notebook, [find-lane-lines.ipynb](./find-lane-lines.ipynb). The project has now been completely refactored and reorganized into modular components to closely match the components in the image processing pipeline. The refactored Python classes are located in the [lane_finder](./lane_finder) subdirectory/module. In [lane_detector.py](./lane_finder/lane_detector.py) you will find the main driver method, `LaneDetector.detect_lane()`. [Using_the_LaneDetector_Class.ipynb](./Using_the_LaneDetector_Class.ipynb) demonstrates how the LaneDetector class is used in detecting the traffic lane in an input image of the road. A [refactored output video](./output_videos/refactored_output.mp4) is also generated in the demo notebook.

The above gif image is a sample of the newly generated output video from the refactored pipeline. The video has been ehanced with additional details in a side panel, displaying the various pipeline stages of the image processing. The details panel include images from:
1. the binary thresholding
2. the image warping into a "birds-eye" view of the road
3. the sliding window detection of lane line pixels
4. a histogram plot of the detected lane line pixels
5. the calculated lane line curvatures in the road

---

### The Project

The steps of this project are:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image to get a "birds-eye" view.
* Detect lane pixels and fit to find the lane boundaries.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of the lane curvature and the vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames.

Examples of the output from each stage of the pipeline are saved in the folder called `ouput_images`. A description of what each image shows is described in the [writeup](./writeup_report.md) for the project. The video called [`project_video.mp4`](./test_videos/project_video.mp4) is the video the pipeline processes. The final pipeline output resulting from processing the test video is the [`project_output.mp4`](./output_videos/project_output.mp4).

The [`challenge_video.mp4`](./test_videos/challenge_video.mp4) video is an extra (and optional) challenge to test the pipeline under somewhat trickier conditions. The resultant video is [`challenge_output.mp4`](./output_videos/challenge_output.mp4).

The [`harder_challenge.mp4`](./test_videos/harder_challenge.mp4) video is another optional challenge and is brutal! My result in [`harder_challenge_output.mp4`](./output_videos/harder_challenge_output.mp4) is not pretty. I leave this one as an exercise for the reader to tweak the pipeline to see how well the pipeline can be improved to get better results :-).

---

### Environment

###### Suggested setup:
* [Anaconda](https://www.anaconda.com/download/) - Python platform for Data Science
* Use conda import of [environmenmt.yml](./environment.txt) to replicate the python environment to run the Jupyter Notebook for this project.
```
    conda env create --file environment.yml
```

The project notebook is [Using_the_LaneDetector_Class.ipynb](./Using_the_LaneDetector_Class.ipynb.ipynb)


###### Note:
1. The original project notebook is [find-lane-lines.ipynb](./find-lane-lines.ipynb). Use the [environment.txt](./environment.txt) file instead to create the conda environment for the older notebook.
2. Because of their large sizes, the project's video files are stored in [Git LFS](https://github.com/git-lfs/git-lfs). Please install the Git LFS client to ensure that video files are downloaded properly when this project repo is cloned.