import cv2 as cv
import numpy as np
import os
##print("file exists?", os.path.exists('IMG_6274.MOV'))
cap = cv.VideoCapture('IMG_6273.MOV')
cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
while cap.isOpened():
    ret, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    ##lower bound of RED HSV: Hue(0-10)
    lower1 = np.array([0, 100, 20])
    upper1= np.array([10, 255, 255])
    

    ##upper boundary of RED HSV; Hue(160-180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    lower_mask = cv.inRange(hsv, lower1, upper1)
    upper_mask = cv.inRange(hsv, lower2, upper2)

    full_mask = cv.bitwise_or(lower_mask,upper_mask)

    res = cv.bitwise_and(frame,frame, mask=full_mask)
    if not ret:
        print("Cant receive frame (stream end?. Exiting ...")
        break
    ##gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_image = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)[1]\
    
    kernel = np.ones((5,5),np.uint8)

    closing = cv.morphologyEx(thresh_image,cv.MORPH_CLOSE, kernel=kernel)

    contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    result = res.copy()
    contours = contours[0] if len(contours) == 2 else contours[1]
    for i in contours:
        x,y,w,h = cv.boundingRect(i)
        cv.rectangle(res,(x,y), (x+w-1,y+h-1),(255,0,0), 4)
        
    cv.namedWindow("resized threshold",cv.WINDOW_FULLSCREEN)
    cv.namedWindow("resized result",cv.WINDOW_FULLSCREEN)
    cv.namedWindow("resized bounding box",cv.WINDOW_FULLSCREEN)
    
    cv.resizeWindow("resized threshold", 1920,1080)
    cv.resizeWindow("resized result", 1920,1080)
    cv.resizeWindow("resized bounding box", 1920,1080)

    
    cv.imshow('resized threshold', thresh_image)
    cv.imshow('resized result',res)
    cv.imshow('resized bounding box', result)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()