 
import cv2
import numpy as np
 
 
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
frameNum = 0 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  frameNum += 1
  if ret == True:   
    tempframe = frame    
    if(frameNum==1):
        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
        print(111)
    if(frameNum>=2):
        currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)        
        currentframe = cv2.absdiff(currentframe,previousframe) 
        median = cv2.medianBlur(currentframe,3)
        
#        img = cv2.imread("E:/chinese_ocr-master/4.png")
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
        gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)
 
        print(222)
 
        # Display the resulting frame
        cv2.imshow('原图',frame) 
        cv2.imshow('Frame',currentframe) 
        cv2.imshow('median',median) 
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(33) & 0xFF == ord('q'):
          break    
    previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
