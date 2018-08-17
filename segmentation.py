import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import imutils
from random import *
from pylab import arange, array, uint8

''' global variables '''
refPt = []
cropping = False
cropped = None
temp = None
clone = None
start_point = None
mouse_down = False

def crop(event, x, y, flags, param):
	''' Listens for click events and saves location of mouse. '''

	global refPt, cropping, cropped, start_point, temp, mouse_down, clone
	temp_image = clone.copy()

	# listens for mouse movement when click is down to draw dynamic rectangle
	# is slow for a second then runs smoothly
	if mouse_down and event == cv2.EVENT_MOUSEMOVE:
		temp = temp_image
		cv2.rectangle(temp, start_point, (x,y), (0, 255, 0), 1)

	# listen for mouse click
	# save (x,y) of mouse at click
	if event == cv2.EVENT_LBUTTONDOWN:
		mouse_down = True
		refPt = [(x, y)]
		if start_point is None:
			start_point = (x, y)
		cropping = True

	# listen for mouse click release
	elif event == cv2.EVENT_LBUTTONUP:
		mouse_down = False
		# save (x, y) of mouse at release
		refPt.append((x, y))
		cropping = False

		# draw rectangle marking roi
		cv2.rectangle(cropped, refPt[0], refPt[1], (0, 255, 0), 2)


def get_gamma_from_brightness(image):
	'''
	Determines good gamma value based on brightness of image
	(dark = large gamma, bright = small gamma).
	'''

	hsv = image.copy()
	hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
	v_hist = cv2.calcHist(images=[hsv], channels=[2], mask=None, histSize=[256], ranges=[0, 256])

	# sort histogram in descending order
	simple_hist = [x[0] for x in v_hist]
	indices = list(range(0, 256))
	sort_hist = [(x, y) for y,x in sorted(zip(simple_hist, indices), reverse=True)]

	max_pixel_color = (sort_hist[0][0] + sort_hist[1][0]+ sort_hist[2][0])/3

	if max_pixel_color in range(0, 75):
		return 3.5
	elif max_pixel_color in range(75, 200):
		return 2.0
	else:
		return 0.75


def adjust_gamma(image, gamma):
	''' Adjusts the gamma levels in image to improve segmentation. '''

	inv_gamma = 1.0/gamma
	table = np.array([(( i/255.0 )**inv_gamma)*255 for i in np.arange(0, 256)]).astype("uint8")
	adjusted_img = cv2.LUT(image, table)
	cv2.imshow('gamma', adjusted_img)
	cv2.waitKey(0)

	return adjusted_img


def increase_contrast(clone):
	''' Increases image contrast to aid in foreground extraction. '''

	image = clone.copy()

	#TODO: more experimentation needed to determine effectiveness
	# calculate gamma value based on brightness of image
	gamma = get_gamma_from_brightness(image)
	# adjust gamma levels in image
	image = adjust_gamma(image, gamma)

	# method 1
	clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l,a,b = cv2.split(lab)
	l2 = clahe.apply(l)
	lab = cv2.merge((l2, a, b))
	image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

	# method 2
	# maxIntensity = 255.0
	# x = arange(maxIntensity)
	# phi = 1
	# theta = 1
	# image = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
	# image = array(image, dtype=uint8)

	cv2.imshow('img', image)
	cv2.waitKey(0)
	return image


def foreground_extraction(image):
	'''	Extracts foreground from roi determined by mouse click coordinates. '''

	global refPt, cropping
	# (y1, y2) (x1, x2)
	points = [(refPt[0][1], refPt[1][1]), (refPt[0][0], refPt[1][0])]
	# define rectangle from mouse click points: (start_x, start_y, width, height)
	rect = (points[1][0], points[0][0], points[1][1]-points[1][0], points[0][1]-points[0][0])

	# masks used by grabcut to mark foreground and background
	mask = np.zeros(image.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	# apply grabcut
	cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	# convert all pixels marked as 0 or 2 into 0 (background)
	# convert all pixels marked as 1 or 3 into 1 (foreground)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	# multiply mask with original image to get cropped roi
	img = image*mask2[:,:,np.newaxis]
	# create a white mask and multiply with roi mask to get roi as a white blob
	# to make the contour process easier
	white_mask = np.ones(image.shape[:2],np.uint8)*255
	white_mask = white_mask[:,:,np.newaxis]*mask2[:,:,np.newaxis]
	return white_mask


def get_roi_pixels(final, foreground):
	'''
	Finds contour of the extracted foreground then gets the pixels within the
	boundary of the contour.
	'''

	# dilate to get smooth blob
	kernel = np.ones((5, 5), np.uint8)
	dilate = cv2.dilate(foreground, kernel, iterations=3)
	erode = cv2.erode(dilate, kernel, iterations=2)

	# get contours of blob (mask)
	cnts,_ = cv2.findContours(erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	maxsize = 0
	best = 0
	count = 0
	for cnt in cnts:
		# single, max contour
	    if cv2.contourArea(cnt) > maxsize:
	        maxsize = cv2.contourArea(cnt)
	        best = count
	    count += 1

	# draw single max contour
	cv2.drawContours(final, cnts, best, (0, 255, 0), 2)

	# get pixels inside contour boundary using another mask
	mask = np.zeros(final.shape, np.uint8)
	cv2.drawContours(mask, [cnts[best]], 0, 255, -1)
	pixelpoints = np.where(mask==255)
	return pixelpoints


def color_roi(clone, pixelpoints, image):
	''' Colors the pixels that lie within the contour boundary. '''

	# add transparent overlay to pixels within contour boundary
	clone[pixelpoints[0], pixelpoints[1]] = [randint(0, 255), randint(0, 255), randint(0, 255)]
	cv2.addWeighted(clone, 0.5, image, 1 - 0.5, 0, image)


def display_results(cropped, image):
	''' Displays final results. '''

	fig, axes = plt.subplots(1, 2, figsize=(8, 4))
	ax = axes.ravel()

	ax[0].set_title('ROI picture')
	ax[0].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
	ax[0].set_axis_off()

	ax[1].set_title('Segmented picture')
	ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	ax[1].set_axis_off()

	plt.tight_layout()
	plt.show()


def main():
	global cropped, temp, clone
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to the image")
	args = vars(ap.parse_args())

	# load the image, and resize if necessary
	final_image = cv2.imread('test_imgs/' + args["image"])
	height, width, _ = final_image.shape
	if height > 600 or width > 700:
		final_image = imutils.resize(final_image, height=600, width=700)

	# clone image and setup the mouse callback function
	# multiple clones needed for different steps
	clone = final_image.copy()
	temp = final_image.copy()
	original = final_image.copy()
	cropped = final_image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", crop)

	# keep looping until the 'q' key is pressed
	while True:
		# wait for key press
		cv2.imshow("image", temp)
		key = cv2.waitKey(1) & 0xFF

		# listen for 'r' key to refresh cropping region
		if key == ord("r"):
			temp = clone.copy()

		# listen for 'c' key to break loop and crop roi
		elif key == ord("c"):
			break

	# if two mouse click points were found
	if len(refPt) == 2:
		constrast_img = increase_contrast(clone)

		foreground_img = foreground_extraction(constrast_img)

		pixelpoints = get_roi_pixels(final_image, foreground_img)

		color_roi(clone, pixelpoints, final_image)

		display_results(cropped, final_image)


if __name__ == "__main__":
	main()
