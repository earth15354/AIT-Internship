import cv2 as cv
import numpy as np

img_og = cv.imread('Images/haitai_og.jpeg',0)
img_og2 = cv.imread('Images/haitai_og2.jpeg',0)
img_pic = cv.imread('Images/haitai_pic.jpg',0)

orb = cv.ORB_create(nfeatures=1000)

# kp are key points, des are descriptors
kp_og, des_og = orb.detectAndCompute(img_og, None)
kp_og2, des_og2 = orb.detectAndCompute(img_og2, None)
kp_p, des_p = orb.detectAndCompute(img_pic, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des_og2, des_p, k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.77*n.distance :
        good_matches.append([m])
    
img_matches = cv.drawMatchesKnn(img_og, kp_og, img_pic, kp_p, good_matches, None, flags=2)

cv.imshow('original image', img_og)
cv.imshow('manual image', img_pic)
cv.imshow('matches', img_matches)
cv.waitKey(0)
