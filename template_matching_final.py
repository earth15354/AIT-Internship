import cv2 as cv
import numpy as np
import os

orb = cv.ORB_create(nfeatures=1000)

path = "ImageTemplates"
images = []
names = []
myList = os.listdir(path)

for c in myList:
    imgCur = cv.imread(f'{path}/{c}', 0)
    images.append(imgCur)
    names.append(os.path.splitext(c)[0])

def findDes(images):
    desList = []
    for im in images:
        kp,des = orb.detectAndCompute(im, None)
        desList.append(des)
    return desList

def findID(image, desList, thresh):

    matchCount = []
    finalVal = None

    # kp are key points, des are descriptors
    kp_p, des_p = orb.detectAndCompute(image, None)

    bf = cv.BFMatcher()
    try:
        for des_og in desList:
            matches = bf.knnMatch(des_og, des_p, k=2)

            good_matches = []
            for m,n in matches:
                if m.distance < 0.77*n.distance :
                    good_matches.append([m])
            matchCount.append(len(good_matches))
    except:
        pass
    
    if len(matchCount) != 0:
        if max(matchCount) >= thresh:
            finalVal = matchCount.index(max(matchCount))
    
    return finalVal

capture = cv.VideoCapture(0)

while True:
    # Captures a frame from the webcam
    ret, frame = capture.read()

    if ret == True:
        # Mirror Image
        # frame_flip = cv.flip(frame, 1)

        # Converts to grayscale
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        idVal = findID(grayscale, findDes(images), 22)

        if idVal != None:
            cv.putText(frame, "This is '" + names[idVal] + "'!", (15, 30), cv.FONT_HERSHEY_COMPLEX,1,(200,255,50),2)

        cv.imshow('Webcam', frame)

        # Wait 30 milliseconds and if "q" is pressed, close the window
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        print("was not successful")
        break