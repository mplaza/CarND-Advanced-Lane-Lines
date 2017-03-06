import numpy as np
import cv2
import glob
import pickle

# store 3d object points
objpoints = []
# store 2d image points
imgpoints = []

# create 3D coordinate points
objpts = np.zeros((6*9, 3), np.float32)
objpts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# list calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

#get chessboard corners of images

for indx, img in enumerate(images):
	img = cv2.imread(img)
	# convert to grayscale
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# find chessboard corners
	ret, corners = cv2.findChessboardCorners(gray_img, (9,6), None)

	# if found the corners add them to the img points array. add obj points from above to obj pt array
	if ret == True:
			objpoints.append(objpts)
			imgpoints.append(corners)

			# display the corners
			img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
			corners_img_name = './camera_cal/corners' + str(indx) + '.jpg'
			cv2.imwrite(corners_img_name, img)

# load img to get size for calibration
img = cv2.imread('./camera_cal/calibration1.jpg')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

# save the camera calibration result
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))














