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

with open('imgpoints.pickle', 'rb') as f:
    imgpoints = pickle.load(f)
with open('objpoints.pickle', 'rb') as f:
    objpoints = pickle.load(f)


################
# 2. Distortion correction
################
def undistort(img,objpoints,imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist



def test_undistort():

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
    #warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

    return M, MI


def test_perspective_warp():

    img = cv2.imread('test_images/test4.jpg') #straight_lines1
    # undistort the image
    undist = undistort(img, objpoints, imgpoints)

    M = perspective_warp(undist)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1],undist.shape[0]),flags=cv2.INTER_LINEAR)
    plt.imshow(warped)
    plt.show()

#test_perspective_warp()


########################
#4. Use gradients to create a thresholded binary image
########################

#sobel x transformation
def sobelX_transform(img, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=21)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    thresh_min = thresh[0]
    thresh_max = thresh[1]
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
    #target yellow color
    y_r = 233
    y_g = 198
    y_b = 148

    width = 12
    yellow = np.zeros_like(B)
    yellow[(B>y_b-width)
           &(B<y_b+width)
           &(G>y_g-width)
           &(G<y_g+width)
           &(R>y_r-width)
           &(R<y_r+width)]=1

    # target white color
    w_r = 225
    w_g = 223
    w_b = 228
    white = np.zeros_like(B)
    w_factor = 10
    white[(B > w_b - w_factor)
           & (B < 255)
           & (G > w_g - w_factor)
           & (G < 255)
           & (R > w_r - w_factor)
           & (R < 255)] = 1

    combined = np.zeros_like(B)
    combined[((white == 1) | (yellow == 1)) & (s_binary ==1)] = 1
    return combined


def combine_gradient(img):
    gradx = sobelX_transform(img, thresh=(35, 130))
    #grady = sobelY_transform(img)
    grad_mag = mag_gradient(img, mag_thresh=(33, 255))
    #grad_dir = dir_gradient(img, thresh=(0.6, 2))
    color = color_threshold(img, thresh=(65, 255))

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) | grad_mag == 1) | (color == 1)] = 1
    return combined

def test_threshold():
    #img = cv2.imread('test_images/straight_lines2.jpg')
    img = cv2.imread('test_images/test1.jpg')

    # undistort the image
    undist = undistort(img, objpoints, imgpoints)

    M = perspective_warp(undist)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)

    combined = combine_gradient(warped)

    plt.imshow(combined, cmap='gray')
    plt.show()

#test_threshold()


################
#6. Find the lane. Detect lane pixels and fit to find the lane boundary
################
def find_lane(img, draw = False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    #an output image to draw the results
    out_img = np.dstack((img, img, img)) * 255 #3 channels

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0] / 2) #midpoint of the graph
    leftx_base = np.argmax(histogram[:midpoint]) #left peak
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # right peak

    #number of sliding window
    nwindows = 9

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
    margin = 90
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
    left_fit = np.polyfit(lefty, leftx, 2) # fit polynomial across all selected points
    right_fit = np.polyfit(righty, rightx, 2)


    #print("Left error: {:,}".format(int(left_error)))
    #print("Right error: {:,}".format(int(right_error)))


    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # plot the chart
    if draw == True:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit, leftx, rightx, lefty, righty, left_fitx, right_fitx

def previous_poly(img, left_fit, right_fit, draw = False):
    #after knowing the lane from previous image, just to search within the confined area
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    if draw == True:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit, leftx, rightx, lefty, righty, left_fitx, right_fitx


def test_find_lane():
    #img = cv2.imread('test_images/straight_lines2.jpg')
    img = cv2.imread('test_images/test6.jpg')
    # test4 is a stress test
    # undistort the image
    undist = undistort(img, objpoints, imgpoints)

    M, MI = perspective_warp(undist)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)

    grad_img = combine_gradient(warped)

    left_fit, right_fit = find_lane(grad_img, draw=True)

    #previous_poly(grad_img, left_fit,right_fit, draw=True)


test_find_lane()

#################
#7. Calculate radius of lane curvature
#################

def cal_radius(shape, left_fit, right_fit):
    ploty = np.linspace(0, shape[0] - 1, shape[0])  # to cover same y-range as image
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad



def cal_radius_real(leftx, rightx, lefty, righty):
    y_eval_l = np.max(lefty)  #where radius is evaluated
    y_eval_r = np.max(righty)
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ym_per_pix = 30 / 720  # meters per pixel in y dimension


    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_l * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_r * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    #offset in meters

    left_peak = (left_fit_cr[0]**2) * 720 * ym_per_pix + left_fit_cr[1]*720 * ym_per_pix+left_fit_cr[2]
    right_peak = (right_fit_cr[0] ** 2) * 720 * ym_per_pix + right_fit_cr[1] * 720 * ym_per_pix + right_fit_cr[2]

    offset = 1280/2*xm_per_pix - left_peak - (right_peak-left_peak)/2


    # Now our radius of curvature is in meters
    return left_curverad, right_curverad, offset



def test_cal_radius():
    # img = cv2.imread('test_images/straight_lines2.jpg')
    img = cv2.imread('test_images/test6.jpg')

    # undistort the image
    undist = undistort(img, objpoints, imgpoints)

    M, MI = perspective_warp(undist)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)

    grad_img = combine_gradient(warped)

    left_fit, right_fit, leftx, rightx, lefty, righty= find_lane(grad_img, draw=False)

    #a,b = cal_radius(grad_img.shape, left_fit, right_fit)
    l_m,r_m, offset= cal_radius_real(leftx, rightx, lefty, righty)

    print('Left radius is {:.1f} m'.format(l_m))
    print('Right radius is {:.1f} m'.format(r_m))
    if(offset > 0):
        print('Vehicle is {:.2f} m left from the center'.format(offset))
    elif(offset < 0):
        print('Vehicle is {:.2f} m right from the center'.format(offset))


#test_cal_radius()

################
#8. draw the detected lane on the image
################


def draw_lanes(Minv, left_fitx, right_fitx, undist, dim = (720, 1280), draw = False):
    ploty = np.linspace(0, dim[0] - 1, dim[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros(dim).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (dim[1], dim[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    if draw == True:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result)
        plt.show()

    return result

def test_draw_lanes():
    #img = cv2.imread('test_images/straight_lines2.jpg')
    img = cv2.imread('test_images/test6.jpg')
    # test4 is a stress test
    # undistort the image
    undist = undistort(img, objpoints, imgpoints)

    M, MI = perspective_warp(undist)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)

    grad_img = combine_gradient(warped)

    left_fit, right_fit, leftx, rightx, lefty, righty, left_fitx, right_fitx = find_lane(grad_img, draw=False)

    draw_lanes(MI, left_fitx, right_fitx, undist)

#test_draw_lanes()

################
#9. pipeline
################

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        self.left_fit =None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None

test_img = cv2.imread('test_images/test6.jpg')
M, MI = perspective_warp(test_img)

line = Line()

def lane_pipeline(img):

    failure_rate = 1.3

    undist = undistort(img, objpoints, imgpoints)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1],undist.shape[0]),flags=cv2.INTER_LINEAR)
    grad_img = combine_gradient(warped)


    if line.detected == False:
        left_fit, right_fit, leftx, rightx, lefty, righty, left_fitx, right_fitx = find_lane(grad_img, draw=False)

        #sanity check
        #check curvature
        l_m, r_m, offset = cal_radius_real(leftx, rightx, lefty, righty)
        diff_curverad = abs ((l_m-r_m) / min(abs(l_m),abs(r_m)))

        left_center = np.average(left_fitx)
        right_center = np.average(right_fitx)
        diff_x = right_center - left_center

        # also shall check mid point lane distance

        fail_condition = diff_curverad > failure_rate or diff_x > 860 or diff_x < 500

        #other conditions: 1) curv diff from last image on the same side
        #
        #no properlane detected
        if fail_condition and line.left_fitx != None:
            line.detected = False
            result = draw_lanes(MI, line.left_fitx, line.right_fitx, undist, draw=False)

        else:
            line.detected = True
            line.left_fit = left_fit
            line.right_fit = right_fit
            line.left_fitx = left_fitx
            line.right_fitx = right_fitx
            result = draw_lanes(MI, left_fitx, right_fitx, undist, draw=False)


    elif line.detected == True:
        left_fit, right_fit, leftx, rightx, lefty, righty, left_fitx, right_fitx = previous_poly(grad_img, line.left_fit, line.right_fit, draw=False)

        l_m, r_m, offset = cal_radius_real(leftx, rightx, lefty, righty)
        diff_curverad = abs((l_m - r_m) / min(abs(l_m), abs(r_m)))

        left_center = np.average(left_fitx)
        right_center = np.average(right_fitx)
        diff_x = right_center - left_center

        fail_condition = diff_curverad > failure_rate or diff_x > 860 or diff_x < 500

        if fail_condition:
            line.detected = False
            result = draw_lanes(MI, line.left_fitx, line.right_fitx, undist, draw=False)

        else:
            line.detected = True
            line.left_fit = left_fit
            line.right_fit = right_fit
            line.left_fitx = left_fitx
            line.right_fitx = right_fitx
            result = draw_lanes(MI, left_fitx, right_fitx, undist, draw=False)

    return result




##################
#10. processing video
##################
from moviepy.editor import VideoFileClip



def process_video(video_path):

    #clip1 = VideoFileClip(video_path).subclip(38,39) #challening part at 22 and 39 second
    clip1 = VideoFileClip(video_path)
    video = clip1.fl_image(lane_pipeline)
    video.write_videofile('lane_marking.mp4', audio=False)

#process_video("project_video.mp4")
