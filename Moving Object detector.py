                                                #MOVING OBJECT DETECTION MODEL


import cv2#opencv

import imutils #used in resizing the image

cam=cv2.VideoCapture(0)#to access the camera

firstFrame=None  #Initial frame

area=100 #initializing the area of contour which will be used later

while True: #Infinite Loop
    _,img=cam.read()

    text="normal"

    img=imutils.resize(img,width=500)

    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#Converting to gray image

    gaussianimg=cv2.GaussianBlur(grayimg,(21,21),0)#Converting gaussian blur image

    if firstFrame is None:
        firstFrame=gaussianimg
        
        continue
    
    imgDiff=cv2.absdiff(firstFrame,gaussianimg)

    thresholdimg=cv2.threshold(grayimg,160,255,cv2.THRESH_BINARY)[1] #Converting to threshold image so that the object detection will be easier

    thresholdimg=cv2.dilate(thresholdimg,None,iterations=2)

    contours=cv2.findContours(thresholdimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#Finding contours

    contours=imutils.grab_contours(contours)

    for c in contours:
        if cv2.contourArea(c)<area:# If the area of the contour is less than 100 ignore the contour
            continue
        
        (x,y,w,h)=cv2.boundingRect(c)#determing the coortinates of the contour
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(20,50,25),2)#drawing rectangle to the contour
        
        text="Moving object Detected" #text which should be included in the image/video
        
    print(text)
    
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,155),2)
    
    cv2.imshow("cameraFeed",img)
    
    key=cv2.waitKey(10)
    
    print(key)
    
    if key == ord("c"):#determing the value to stop the video (CAN USE ANY VALUE)
        break
    
cam.release()

cv2.destroyAllWindows()






    
        
            
