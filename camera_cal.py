import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

with open('imgpoints.pickle', 'rb') as f:
    imgpoints = pickle.load(f)
with open('objpoints.pickle', 'rb') as f:
    objpoints = pickle.load(f)

def test_undistort():
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

