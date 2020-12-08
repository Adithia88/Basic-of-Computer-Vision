# -*- coding: utf-8 -*-
import cv2

# read video file
cap = cv2.VideoCapture('videoplayback.mp4')

# count frame
count = 0

# increment frame counter
def getFrameNumber(count):
	count = count + 1
	return count

while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()

	# if frame exists
	if ret == True:
		# Display results
		cv2.imshow("Results", frame)

		# save current frame
		cv2.imwrite("Data-Video2Frame/Data8/frame-%d.jpg"%getFrameNumber(count), frame)

		# increment counter
		count = count + 1
	else:
		break

	# ESC to stop algorithm
	key = cv2.waitKey(7) % 0x100
	if key == 27:
		break

# Clear all windows
cv2.destroyAllWindows()