import numpy as np
from picamera import PiCamera
from collections import deque
from math import sqrt
from datetime import datetime, timedelta
import time
from threading import Timer
import cv2

def distance(point1, point2):
    point1 = list(point1)
    point2 = list(point2)
    dst = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dst

def clearTracker():
    print("cleannnn\n")
    global timerRunning, tracker
    timerRunning = False
    tracker.clear()  


#init the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 40
#init the numpy array for photo to opencv
rawCapture = np.empty((480, 640, 3), dtype=np.uint8) 

#allow the delay of camera 
time.sleep(0.1)
#MOG algorithm is simple but faster then MOG2
#bgsub = cv2.bgsegm.createBackgroundSubtractorMOG()
bgsub = cv2.createBackgroundSubtractorMOG2()
#blob detectorA
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 150
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 0.5 
params.maxArea = 50
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = True
params.minConvexity=0.5
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)

#kernel = np.ones((5,5),np.uint8)

#path track
tracker = deque(maxlen=32)

#config
TRACE_THICKNESS = 5

#timer for clear tracker
timerRunning = False
#emphasize the highlight
whiteLowerBd  = (149, 149, 149)
whiteUpperBd= (255, 255, 255)
#Real-time tracking
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    #cv2.imshow("Origin", frame)
    #apply MOG
    frame_filter = cv2.inRange(frame, whiteLowerBd, whiteUpperBd)
    frame_mask = bgsub.apply(frame_filter)
    #tophat = cv2.morphologyEx(frame_mask, cv2.MORPH_TOPHAT, kernel)
    #cv2.imshow("Frame", frame)
    
    blob = detector.detect(frame_mask)
    #circle = cv2.HoughCircles(frame_mask, cv2.HOUGH_GRADIENT,3,100,param1=100,param2=30,minRadius=4,maxRadius=15)

    frame_hlight = cv2.drawKeypoints(frame_mask, blob, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #if circle is not None:
    #    circle = np.round(circle[0,:]).astype("int")
    #    for (x, y, r) in circle:
    #        cv2.circle(frame_hlight, (x,y),r, (0,255,0),2)
    cv2.imshow("After", frame_hlight)

    #draw blob path
    if len(blob) > 0:
        if len(tracker) == 0 or distance(tracker[0], blob[0].pt) < 50:
            tracker.appendleft((int(blob[0].pt[0]),int(blob[0].pt[1])))
        if len(tracker) > 0 and timerRunning == False:
            print("Timing......\n")
            t = Timer(5, clearTracker)
            t.start()
            timerRunning = True
    tracker_copy = tracker.copy()
    for i in range(1,len(tracker_copy)):
        if tracker_copy[i -1] is None or tracker_copy[i] is None:
            continue
        thickness = int(np.sqrt(32/float(i+1))*2.5)
        cv2.line(frame, tracker_copy[i-1], tracker_copy[i], (0,0,255), thickness)
    cv2.imshow("Origin", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.close()
cv2.destroyAllWindows()
