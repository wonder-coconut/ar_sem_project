import cv2
import numpy as np

videocap = cv2.VideoCapture(0)
videocap.set(3,1280)
videocap.set(4,720)

imgTarget = cv2.imread('../assets/images/watchmen.jpg')
scale = 50
width = int(imgTarget.shape[1] * scale / 100)
height = int(imgTarget.shape[0] * scale / 100)
dsize = (width,height)
imgTarget = cv2.resize(imgTarget, dsize)

myVid = cv2.VideoCapture('../assets/videos/Inglourious_Basterds_Intro.mp4')

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT,hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget,None)


while True:
    success, imgWebCam = videocap.read()
    kp2, des2 = orb.detectAndCompute(imgWebCam,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)
    
    print(len(goodMatches))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebCam, kp2, goodMatches, None, flags=2)
    
    if(len(goodMatches) > 20) :
        srcPts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)
        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebCam,[np.int32(dst)],True,(255,0,0),3)

    cv2.imshow('img2',img2)
    cv2.imshow('imgFeatures', imgFeatures)
    #cv2.imshow('webcam', imgWebCam)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break