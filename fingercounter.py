import cv2 
import numpy as np
import time
import HandTrackerModule as htm
import math
import os
#####################################################
wCam , hCam = 1280 ,720 
#####################################################
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)

folderpath = "fingerphoto"
mylist = os.listdir(folderpath)
print(mylist)
overlaylist = []
for impath in mylist:
    image = cv2.imread(f'{folderpath}/{impath}')
    #print(f'{folderpath}/{impath}')
    overlaylist.append(image)
ptime = 0
detector = htm.HandDetector(detectCon=0.75)

tipIds = [4, 8, 12, 16, 20]
while True :
    success , img = cam.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # For the right hand only , have to change cordinates for the left hand
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # It is for all the rest 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        # Putting the image of the number of hands with images we provided with the webcam image
        h, w, c = overlaylist[totalFingers - 1].shape
        img[0:h, 0:w] = overlaylist[totalFingers - 1]

        
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}', (600, 30), cv2.FONT_HERSHEY_DUPLEX, 1,
                (255, 0, 255), 1)             
    cv2.imshow("image",img)
    cv2.waitKey(1)
