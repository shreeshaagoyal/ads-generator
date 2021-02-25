from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# file path to video to read
videoPath = "C:\\Users\\shree\\Documents\\src\\test\\video-editing\\videos\\1.mp4"

# minimum size in pixels for a region of image to be considered motion
minArea = 500

vs = cv2.VideoCapture(videoPath)

# model the background of videousing only the first frame of video
firstFrame = None

motionDetectionFrame = None

while True:
    frame = vs.read()
    frame = frame[1]
    text = "No motion detected"

    if frame is None:
        break
    
    # PROCESS FRAME AND PREPARE IT FOR MOTION ANALYSIS
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # apply Gaussian smoothing to avg pixel intensities across a 21x21 region

    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute absolute difference between current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 25, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations = 2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if motionDetectionFrame is None:
        motionDetectionFrame = thresh

    # loop over contours
    for c in cnts:
        # if contour is too small, ignore it
        if cv2.contourArea(c) < minArea:
            continue
        
        # compute bounding box for contour, draw it on the frame, and update text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion detected"

    motionDetectionFrame = cv2.addWeighted(motionDetectionFrame, 0.95, thresh, 0.05, 0)

    # draw text and timestamp on the frame
    cv2.putText(frame, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Aggregate Motion Feed", motionDetectionFrame)
    key = cv2.waitKey(1) & 0xFF

    # if 'q' key is pressed, break from loop
    if key == ord("q"):
        break

cv2.imwrite("aggregateMotion.jpg", motionDetectionFrame)

# cleanup
vs.release()
cv2.destroyAllWindows()