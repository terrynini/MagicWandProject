import numpy as np
from picamera import PiCamera
import time
import cv2

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
params.minArea = 10
params.maxArea = 500
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = True
params.minConvexity=0.5
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)


kernel = np.ones((5,5),np.uint8)
#Real-time tracking
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    cv2.imshow("Origin", frame)
    #apply MOG
    frame_mask = bgsub.apply(frame)
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
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


camera.close()
cv2.destroyAllWindows()
