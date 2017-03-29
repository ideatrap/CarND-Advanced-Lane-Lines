

#TODO Use color transforms, gradients, etc., to create a thresholded binary image.
#TODO Apply a perspective transform to rectify binary image ("birds-eye view").
#TODO Detect lane pixels and fit to find the lane boundary.
#TODO Determine the curvature of the lane and vehicle position with respect to center.
#TODO Warp the detected lane boundaries back onto the original image.
#TODO Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import camera_cal

camera_cal.camera_distortion_parameter()

