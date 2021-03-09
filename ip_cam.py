import cv2 
import time
import numpy as np
url = 'http://26.146.143.10:8080/video'
cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(0)
_frames_per_batch_ = 20


show = 0  
while(True):
    images = []
    for count in range(0,20):
        if count <=_frames_per_batch_ and show == 0 : 
            ret, frame = cap.read()
            images.append(frame)
            print('frame:',count,'/20')
    else:
        for frame in images:
            if frame is not None:
                cv2.imshow('frame',frame)
                time.sleep(0.05)
                q = cv2.waitKey(1)
                if q == ord("q"):
                    break
        count = 0
        images = []
cv2.destroyAllWindows()