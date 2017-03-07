import numpy as np
import cv2
import glob
import pickle
from line_tracker import tracker
import matplotlib.pyplot as plt

# read the camera calibration vals

dist_pickle = pickle.load( open( "calibration_pickle.p", "rb") )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# apply sobel operator for gradient theresholding

def threshold_sobel_x(img):
	# define threshold
	thresh = (20,100)
	# convert img to grayscale bc sobel require one color channel
	gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# calc derivative in x direction
	sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
	sobelx_abs = np.absolute(sobelx)
	# convert to 8-bit
	scaled_sobelx = np.uint8(255*sobelx_abs/np.max(sobelx_abs))
	# select pixels based on threshold
	sx_binary = np.zeros_like(scaled_sobelx)
	sx_binary[(scaled_sobelx >= thresh[0]) & (scaled_sobelx <= thresh[1])] = 1
	return sx_binary


def threshold_sobel_y(img):
	# define threshold
	thresh = (30,100)
	# convert img to grayscale bc sobel require one color channel
	gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# calc derivative in y direction
	sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
	sobely_abs = np.absolute(sobely)
	# convert to 8-bit
	scaled_sobely = np.uint8(255*sobely_abs/np.max(sobely_abs))
	# select pixels based on threshold
	sy_binary = np.zeros_like(scaled_sobely)
	sy_binary[(scaled_sobely >= thresh[0]) & (scaled_sobely <= thresh[1])] = 1

	return sy_binary

# color thresholding

def threshold_color(img):
	# convert to hls color space -- works better in diff light conditions
	hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	# define threshold
	s_thresh = (90,255)
	s_channel = hls_img[:,:,2]
	h_channel = hls_img[:,:,0]
	s_binary = np.zeros_like(s_channel)
	s_binary[( (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1]) )] = 1

	# use r channel also -- does well with white lines
	rgb_img = img
	r_thresh = (50,255)
	r_channel = rgb_img[:,:,0]
	r_binary = np.zeros_like(r_channel)
	r_binary[( (r_channel > r_thresh[0]) & (r_channel <= r_thresh[1]) )] = 1

	# combine color thresholds
	output = np.zeros_like(r_channel)
	output[(s_binary == 1) & (r_binary == 1)] = 1
	return output
# load test images
images = glob.glob('./test_images/test*.jpg')
# print out undistorted calibration image to show calibration done correctly
calibration_img_example = cv2.imread('./camera_cal/calibration1.jpg')
undistort_calibration_img = cv2.undistort(calibration_img_example, mtx, dist, None, mtx)
undistorted_img_name = './test_images/undistorted_calibration1.jpg'
cv2.imwrite(undistorted_img_name, undistort_calibration_img)

for indx, img in enumerate(images):
	img = cv2.imread(img)
	# undistort the image using the calibration data
	img = cv2.undistort(img, mtx, dist, None, mtx)

	# process image and apply thresholding to get binary result
	process_image = np.zeros_like(img[:,:,0])

	color_binary = threshold_color(img)
	sobelx_bin = threshold_sobel_x(img)
	sobely_bin = threshold_sobel_y(img)

	# combine different thresholds to get binary result
	process_image[((sobelx_bin == 1) & (sobely_bin == 1) | (color_binary == 1) )] = 255

	# define perspective transformation
	# select trapezoid points of lane line for src
	trapezoid_top_width = .08
	trapezoid_bottom_width = 0.75
	trapezoid_top_height = .61
	trapezoid_bottom_height = .94

	top_right = [ (img.shape[1]*(0.5+trapezoid_top_width/2)), (img.shape[0]*trapezoid_top_height)]
	bottom_right = [ (img.shape[1]*(0.5+trapezoid_bottom_width/2)), (img.shape[0]*trapezoid_bottom_height)]
	bottom_left = [ (img.shape[1]*(0.5-trapezoid_bottom_width/2)), (img.shape[0]*trapezoid_bottom_height)]
	top_left = [ (img.shape[1]*(0.5-trapezoid_top_width/2)), (img.shape[0]*trapezoid_top_height)]


	src = np.float32([top_right, bottom_right, bottom_left, top_left])
	offset = img.shape[1] * 0.22

	# set points for dst img
	dst = np.float32([ [img.shape[1] - offset, 0], [img.shape[1] - offset, img.shape[0]], [offset, img.shape[0]], [offset, 0]])

	# compute perspective transform
	M = cv2.getPerspectiveTransform(src, dst)
	# to inverse the transform later
	Minv = cv2.getPerspectiveTransform(dst, src)
	# get warped image
	warped = cv2.warpPerspective(process_image, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

	# initialize tracking class
	window_width = 50
	window_height = 80
	margin = 25
	# approximate pixel to real world conversions based on lane width 3.7m and lane length of 30m
	ym = 30/720
	xm = 3.7/700
	smooth_factor = 15

	curve_centers = tracker(win_width = window_width, win_height=window_height, margin=margin, ym=ym, xm=xm, smooth_fac=smooth_factor)
	
	# find window centroids
	window_centroids = curve_centers.find_window_centroids(warped)

	# store points to draw curves
	l_pts = np.zeros_like(warped)
	r_pts = np.zeros_like(warped)

	# collect points to find the left and right lane
	leftx = []
	rightx = []

	def window_mask(width, height, img_ref, center,level):
 		output = np.zeros_like(img_ref)
 		output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
 		return output

	# Go through each level and draw the windows 	
	for level in range(0,len(window_centroids)):
		# Window_mask is a function to draw window areas
		l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
		r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
		# Add graphic points from window mask here to total pixels found 
		l_pts[(l_pts == 255) | ((l_mask == 1) ) ] = 255
		r_pts[(r_pts == 255) | ((r_mask == 1) ) ] = 255

		leftx.append(window_centroids[level][0])
		rightx.append(window_centroids[level][1])

  # Draw the results
	template = np.array(r_pts+l_pts,np.uint8) # add both left and right window pixels together
	zero_channel = np.zeros_like(template) # create a zero color channle 
	template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
	warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
	output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

	# fit a curve to the lane lines
	# yvals are determined by the window height because span whole image through levels but x vals by the position of lanes
	yrange = np.linspace(0, img.shape[0]-1, num=img.shape[0])
	# create y vals for all the slices
	slc_yvals = np.arange(img.shape[0]-(0.5*window_height), 0, -window_height)

	# fit second order polynomial to boxes along curves
	left_fit = np.polyfit(slc_yvals, leftx, 2)
	left_fitx = np.int32(np.array(left_fit[0]*yrange**2 + left_fit[1]*yrange + left_fit[2]))
	right_fit = np.polyfit(slc_yvals, rightx, 2)
	right_fitx = np.int32(np.array(right_fit[0]*yrange**2 + right_fit[1]*yrange + right_fit[2]))


	lane = np.zeros_like(img)
	lane_dark = np.zeros_like(img)

	# generate the line points by zipping together the x and y vals for the polynomial
	left_lane_line = np.array(list(zip(left_fitx, yrange)), np.int32)
	right_lane_line = np.array(list(zip(right_fitx, yrange)), np.int32)

	cv2.fillPoly(lane, [left_lane_line], color=[0,0,255])
	cv2.fillPoly(lane, [right_lane_line], color=[0,255,255])
	# make darker so you can see lane line on image
	cv2.fillPoly(lane_dark, [left_lane_line], color=[255,255,255])
	cv2.fillPoly(lane_dark, [right_lane_line], color=[255,255,255])

	# visualize the lane lines over the image 
	# reverse the unwarping 
	lane_warped = cv2.warpPerspective(lane, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	lane_warped_dark = cv2.warpPerspective(lane_dark, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

	polynomial_lane_img = cv2.addWeighted(img, 1.0, lane_warped_dark, -1.0, 0.0)

	# calculate where vehicle is located with respect to the center of the lane
	# find camera center
	camera_center = 0.5*left_fitx[-1] + 0.5*right_fitx[-1]
	vehicle_offset = (camera_center - 0.5*img.shape[1])*xm
	vehicle_side = 'center'
	if vehicle_offset < 0:
		vehicle_side = 'right'
	elif vehicle_offset > 0:
		vehicle_side = 'left'
	vehicle_offset = abs(round(vehicle_offset,2))

	cv2.putText(polynomial_lane_img, 'Car is ' + str(vehicle_offset) + 'm to the ' + vehicle_side, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

	# do a polyfit with the perspective transformed so can get the curvature in meters instead of pixels
	# just use the left line for now since they should be parallel
	yvals_m = np.array(slc_yvals, np.float32)*ym
	leftx_m = np.array(leftx, np.float32)*xm
	polyfit_m = np.polyfit(yvals_m, leftx_m, 2)
	curve_radius = ((1 + (2*polyfit_m[0]*yvals_m[-1] + polyfit_m[1])**2)**1.5) / np.absolute(2*polyfit_m[0])

	cv2.putText(polynomial_lane_img, 'Curve radius is ' + str(abs(round(curve_radius,2))) + 'm', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)



	processed_img_name = './test_images/lane_poly_img' + str(indx) + '.jpg'
	cv2.imwrite(processed_img_name, polynomial_lane_img)



