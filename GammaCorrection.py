import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma=3.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)


video = cv.VideoCapture('Night Drive - 2689.mp4')
scale=0.4
frame_width = int(video.get(3)*scale*2)
frame_height = int(video.get(4)*scale)
# print(frame_width)
# print(frame_height)
out1 = cv.VideoWriter('Problem1Output.mp4', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))
result_display = None

while True:
	ret, main_frame = video.read()
	if ret is None or ret is False:
		print("Cant read")
		break
	mf_original = main_frame.copy()
	result = cv.GaussianBlur(main_frame,(5,5),0)
	result = adjust_gamma(result)
	mf_original_display = cv.resize(mf_original, dsize=None, fx=scale, fy=scale)
	result_display = cv.resize(result, dsize=None, fx=scale, fy=scale)
	image_display = np.concatenate((np.concatenate((result_display, mf_original_display), axis=1),),axis=0)
	cv.imshow('output',image_display)
	out1.write(image_display)
	cv.waitKey(1)

video.release()
cv.destroyAllWindows()
if result_display is not None:
	plt.hist(result_display.ravel(), 256, [0, 256])
	plt.show()
