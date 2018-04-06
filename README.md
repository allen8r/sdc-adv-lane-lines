# Advanced Lane Finding

<p>
<img width="480" height="270" src="./output_images/project_output.gif" alt="Detected Lane">
</p>

The goal of this project is to write a software pipeline to identify the lane boundaries in a traffic video from the perspective of a moving vehicle from which the video was captured. A [writeup](./writeup_report.md) of the project describes the details of the lane detection pipeline.  

---

### The Project

The steps of this project are:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image to get a "birds-eye view".
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

Suggested setup:
* [Anaconda](https://www.anaconda.com/download/) - Python platform for Data Science
* Use conda import of [`environmenmt.txt`](./environment.txt) to replicate the python environment to run the Jupyter Notebook for this project.
```
    conda env create --file environment.txt
```

The project notebook is [`find-lane-lines.ipynb`](./find-lane-lines.ipynb)
