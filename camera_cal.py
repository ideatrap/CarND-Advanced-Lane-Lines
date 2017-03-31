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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    '''

    #undistort the image
    with open('imgpoints.pickle', 'rb') as f:
        imgpoints = pickle.load(f)
    with open('objpoints.pickle', 'rb') as f:
        objpoints = pickle.load(f)
    undist = undistort(img,objpoints,imgpoints)
    '''

    #position the corners
    b_y = int((1 - 0.066) * img.shape[0])  # y of bottom of polygon, remove the bottom 7.77%
    t_y = int((1 - 0.34) * img.shape[0])  # y of top of polygram, remove the top 64.17%
    t_x_l = img.shape[1] / 2 - 92  # x of top left point, 95 px from the middle of the x dim
    t_x_r = img.shape[1] / 2 + 92  # x of top right point, 95 px from the middle of the x dim
    o_height = b_y - t_y  # height of the orginal cropped image
    b_x_l = t_x_l - int(o_height / np.tan(32.2 * pi / 180))#x of the left bottom point
    b_x_r = t_x_r + int(o_height / np.tan(30.5 * pi / 180))#y of the right bottom point

    b_x_l = img.shape[1] / 2 - 420
    b_x_r = img.shape[1] / 2 + 420
    #points of original warped imgae
    # top_left, bottom_left, top_right, bottom_right
    src = np.float32([[t_x_l, t_y], [b_x_l, b_y], [t_x_r, t_y], [b_x_r, b_y]])


    # point of the destination image
    # top_left, bottom_left, top_right, bottom_right
    factor=0.8
    dst = np.float32([[b_x_l, t_y * factor], [b_x_l, b_y], [b_x_r, t_y * factor], [b_x_r, b_y]])


    #perform perspective transformation
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

    return warped


def test_perspective_warp():
    img = cv2.imread('test_images/straight_lines2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warped = perspective_warp(img)
    plt.imshow(warped)
    plt.show()

#test_perspective_warp()


########################
#4. Use color transforms, gradients, etc., to create a thresholded binary image
########################

