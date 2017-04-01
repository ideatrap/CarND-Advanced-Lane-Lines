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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #position the corners
    b_y = int((1 - 0.066) * img.shape[0])  # y of bottom of polygon, remove the bottom 7.77%
    t_y = int((1 - 0.34) * img.shape[0])  # y of top of polygram, remove the top 64.17%
    t_x_l = img.shape[1] / 2 - 92  # x of top left point, 95 px from the middle of the x dim
    t_x_r = img.shape[1] / 2 + 92  # x of top right point, 95 px from the middle of the x dim

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
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

    return warped


def test_perspective_warp():

    img = cv2.imread('test_images/straight_lines1.jpg')

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



def test_gradient_transform():
    img = cv2.imread('test_images/test3.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # undistort the image
    with open('imgpoints.pickle', 'rb') as f:
        imgpoints = pickle.load(f)
    with open('objpoints.pickle', 'rb') as f:
        objpoints = pickle.load(f)
    undist = undistort(img, objpoints, imgpoints)

    warped = perspective_warp(undist)

    gradx= sobelX_transform(warped)
    grady = sobelY_transform(warped)

    grad_max = mag_gradient(warped, mag_thresh=(33,255))
    # direction gradient is not very useful
    #grad_dir = dir_gradient(warped,thresh=(0.6, 2))

    output = np.zeros_like(gradx)
    output[((gradx == 1) & (grady == 1)) | ((grad_max == 1))] = 1

    plt.imshow(output, cmap='gray')
    plt.show()

#test_gradient_transform()


######################
#5. color thresholding
######################

def color_threshold(img):
    pass

def test_color_threshold():
    pass

test_color_threshold()