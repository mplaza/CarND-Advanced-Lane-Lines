import numpy as np
import cv2

class tracker():

	def __init__(self, win_width, win_height, margin, ym, xm, smooth_fac):
		# list for storing center values
		self.centers = []

		# window pixel dimensions of center vals to determine curve
		self.window_width = win_width
		self.window_height = win_height

		self.margin = margin

		# meters per pixel
		self.ym_per_pix = ym
		self.xm_per_pix = xm

		self.smooth_factor = smooth_fac


	# apply sliding window approach with a convolution like in lesson
	def find_window_centroids(self, warped):

		# set window width/ height/ margin
		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin
	    
	    # store window centroid positions
		window_centroids = [] 
		# convolution templace for window
		window = np.ones(window_width) 

	    # get the vertical image slice and then get starting positions for lanes then do 1D convolution the vertical image slice
	    # take warped image and pick slice and squash it together to see pixel density
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
		r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
	    
	  # collect centroid positions
		window_centroids.append((l_center,r_center))
	    
		# loop through the other slices
		for level in range(1,(int)(warped.shape[0]/window_height)):
			# convolve the window into the vertical slice of the image
			image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
			conv_signal = np.convolve(window, image_layer)
			# Find the best left centroid by using past left center as a reference
			# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
			offset = window_width/2
			# use padding to consider region of convolved signal
			l_min_index = int(max(l_center+offset-margin,0))
			l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
			# gets max pixel density per local region
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
			# Find the best right centroid by using past right center as a reference
			r_min_index = int(max(r_center+offset-margin,0))
			r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
			# add to centriod list for that layer
			window_centroids.append((l_center,r_center))
		# add to the line_tracker recent centers
		self.centers.append(window_centroids)
		# use the smooth factor to average line center values and prevent marker from jumping around
		return np.average(self.centers[-self.smooth_factor:], axis = 0)
