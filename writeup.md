

# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.




### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   
This is the document.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the section 1 of `model.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text](https://s25.postimg.org/ljhh5ycxr/Screen_Shot_2017-03-29_at_10.42.45_PM.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text](https://s25.postimg.org/f72bw49vj/Screen_Shot_2017-03-29_at_11.37.46_PM.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. The codes are contained in section 5 of `model.py`.

The most useful transformation is gradient X and S channel. I have also used RGB channel to select a range of yellow and white color to narrow down the lane selection

Here's an example of my output for this step.  

![alt text](https://s25.postimg.org/72u7rdngf/channel_S_threading.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_warp()`, which appears in section 3 in the file `model.py`. The `perspective_warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```
b_y = int((1 - 0.08) * img.shape[0])  # y of bottom of polygon, remove the bottom 7.77%
t_y = int((1 - 0.34) * img.shape[0])  # y of top of polygram, remove the top 64.17%
t_x_l = img.shape[1] / 2 - 95
t_x_r = img.shape[1] / 2 + 96
b_x_l = img.shape[1] / 2 - 420
b_x_r = img.shape[1] / 2 + 420

src = np.float32([[t_x_l, t_y], [b_x_l, b_y], [t_x_r, t_y], [b_x_r, b_y]])

factor = 0.5
dst = np.float32([[b_x_l, t_y * factor], [b_x_l, b_y], [b_x_r, t_y * factor], [b_x_r, b_y]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 545, 475      | 220, 237        |
| 220, 662      | 220, 662      |
| 1060, 662     | 1060, 662      |
| 735, 475      | 1060, 237        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text](https://s25.postimg.org/csf1pfj0f/Screen_Shot_2017-03-31_at_10.17.35_PM.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial in section 6 of `model.py`

![alt text](https://s25.postimg.org/qaly1pv5r/Screen_Shot_2017-04-07_at_11.17.01_PM.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in section 7 of `model.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in section 8 in `model.py` in the function `draw_lanes()`.  Here is an example of my result on a test image:

![alt text](https://s25.postimg.org/grc9897nj/Screen_Shot_2017-04-07_at_11.19.20_PM.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/jV1eY3ZL__U)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of the project is to detect the lane through various threshold and color picking. It's most sensitive to the marking on the road.

If the color contrast appears hard to recognize, it will put the lane onto abrupt change. In this case, it relies on the logic to validate the validity of the lane, and decide whether to use the lane in current frame, or inherent from previous frame.

Even the lane is recognized, it needs a good smoothing filter so that the car doesn't make sudden turn.

To make things worse, if the lanes are hard to recognize, and at the same time, it's making sharp turn. It's quite dangerous to blindly inherent previous lanes. If the car is traveling at high speed, it may rush out of the road. In this case, the car shall be instructed to slow down so that it has more frames to use to recognize the lanes.
