import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import pi

#########
#1. function to obtain camera distortion parameters
#########
def camera_distortion_parameter():
# obtain the camera distortion parameter
    image_list = 20
    i = 1
    nx = 9 #9 corners horizontally
    ny = 6 #6 corners vertically

    objpoints = [] # Coordinate of the real chessboard  corners
    imgpoints = [] # Coordinate of the chessboard corners from camera images


    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    while i <= image_list:
        path = 'camera_cal/calibration' + str(i) + '.jpg'
        # with open(path,'rb') as file:
        img = cv2.imread(path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #find the corner of the image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        #if corners are identified
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            #display the corner
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        i += 1
    with open('imgpoints.pickle', 'wb') as f:
        pickle.dump(imgpoints,f)
    with open('objpoints.pickle', 'wb') as f:
        pickle.dump(objpoints,f)


#camera_distortion_parameter()


################
# 2. Distortion correction
################
def undistort(img,objpoints,imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist



def test_undistort():
    with open('imgpoints.pickle', 'rb') as f:
        imgpoints = pickle.load(f)
    with open('objpoints.pickle', 'rb') as f:
        objpoints = pickle.load(f)

    #img = cv2.imread('camera_cal/calibration4.jpg')
    img = cv2.imread('test_images/test1.jpg')
    plt.imshow(img)
    plt.show()


    undist = undistort(img, objpoints, imgpoints)
    plt.imshow(undist)
    plt.show()

#test_undistort()

##########################
# 3. create bird eye view through perspective transformation
##########################

def perspective_warp(img):
    #position the corners
    b_y = int((1 - 0.08) * img.shape[0])  # y of bottom of polygon, remove the bottom 7.77%
    t_y = int((1 - 0.34) * img.shape[0])  # y of top of polygram, remove the top 64.17%
    t_x_l = img.shape[1] / 2 - 95  # x of top left point, 95 px from the middle of the x dim
    t_x_r = img.shape[1] / 2 + 96  # x of top right point, 95 px from the middle of the x dim

    b_x_l = img.shape[1] / 2 - 420 # x of bottom left
    b_x_r = img.shape[1] / 2 + 420 # x of bottom right
    #points of original warped imgae
    # top_left, bottom_left, top_right, bottom_right
    src = np.float32([[t_x_l, t_y], [b_x_l, b_y], [t_x_r, t_y], [b_x_r, b_y]])


    # point of the destination image
    # top_left, bottom_left, top_right, bottom_right
    factor=0.5
    dst = np.float32([[b_x_l, t_y * factor], [b_x_l, b_y], [b_x_r, t_y * factor], [b_x_r, b_y]])


    #perform perspective transformation
    M = cv2.getPerspectiveTransform(src, dst)
    MI = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

    return warped


def test_perspective_warp():

    img = cv2.imread('test_images/test4.jpg') #straight_lines1
    # undistort the image
    with open('imgpoints.pickle', 'rb') as f:
        imgpoints = pickle.load(f)
    with open('objpoints.pickle', 'rb') as f:
        objpoints = pickle.load(f)
    undist = undistort(img, objpoints, imgpoints)

    warped = perspective_warp(undist)
    plt.imshow(warped)
    plt.show()

#test_perspective_warp()


########################
#4. Use gradients to create a thresholded binary image
########################

#sobel x transformation
def sobelX_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=19)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    thresh_min = 20
    thresh_max = 110
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def sobelY_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=19)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    thresh_min = 20
    thresh_max = 110
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def mag_gradient(img, sobel_kernel=19, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1


    return binary_output


def dir_gradient(img, sobel_kernel=19, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    #Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    #Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dirG = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dirG)
    binary_output[(dirG >= thresh[0]) & (dirG <= thresh[1])] = 1

    return binary_output



######################
#5. color thresholding
######################

def color_threshold(image, thresh = (0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    S = hls[:, :, 2]
    s_binary = np.zeros_like(S)
    s_binary[(S >= thresh[0]) & (S <= thresh[1])] = 1

    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    #select yellow
    #target yellow range
    y_r = 233
    y_g = 198
    y_b = 108

    yellow = np.zeros_like(B)
    yellow[(B>y_b-80)
           &(B<y_b+60)
           &(G>y_g-80)
           &(G<y_g+60)
           &(R>y_r-28)
           &(R<y_r+28)]=1

    # target white, bigger the number, whiter
    w_r = 225
    w_g = 223
    w_b = 228
    white = np.zeros_like(B)
    white[(B > w_b - 80)
           & (B < 255)
           & (G > w_g - 80)
           & (G < 255)
           & (R > w_r - 28)
           & (R < 255)] = 1

    combined = np.zeros_like(B)
    combined[((white == 1) | (yellow == 1)) &(s_binary ==1)] = 1
    return s_binary


def combine_gradient(img):
    gradx = sobelX_transform(img)
    # grady = sobelY_transform(warped)
    grad_mag = mag_gradient(img, mag_thresh=(33, 255))
    #grad_dir = dir_gradient(img, thresh=(0.6, 2))
    color = color_threshold(img, thresh=(65, 255))

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) | grad_mag == 1) & (color == 1)] = 1
    return combined

def test_threshold():
    #img = cv2.imread('test_images/straight_lines2.jpg')
    img = cv2.imread('test_images/test2.jpg')
    #test4 is a stress test

    # undistort the image
    with open('imgpoints.pickle', 'rb') as f:
        imgpoints = pickle.load(f)
    with open('objpoints.pickle', 'rb') as f:
        objpoints = pickle.load(f)
    undist = undistort(img, objpoints, imgpoints)

    warped = perspective_warp(undist)

    combined = combine_gradient(warped)

    plt.imshow(combined, cmap='gray')
    plt.show()

#test_threshold()


################
#6. Find the lane
################
def find_lane(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    #an output image to draw the results
    out_img = np.dstack((img, img, img)) * 255 #3 channels

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0] / 2) #midpoint of the graph
    leftx_base = np.argmax(histogram[:midpoint]) #left peak
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # right peak

    #9 sliding window
    nwindows = 2

    window_height = np.int(img.shape[0] / nwindows) #y axis for each window

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    #current position is the peak of the histogram
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 57
    # Set minimum number of pixels found to recenter window
    minpix = 45

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []



    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)

        #moving from top to bottom
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2) # fit polynomial across all visible points
        right_fit = np.polyfit(righty, rightx, 2)



    # plot the chart
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def test_find_lane():
    img = cv2.imread('test_images/straight_lines1.jpg')
    #img = cv2.imread('test_images/test2.jpg')
    # test4 is a stress test
    # undistort the image
    with open('imgpoints.pickle', 'rb') as f:
        imgpoints = pickle.load(f)
    with open('objpoints.pickle', 'rb') as f:
        objpoints = pickle.load(f)
    undist = undistort(img, objpoints, imgpoints)

    warped = perspective_warp(undist)

    combined = combine_gradient(warped)

    find_lane(combined)

test_find_lane()
