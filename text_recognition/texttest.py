import pytesseract
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import re
import sys

def imageTextDetection(imagePath='../assets/images/image1.jpg'):
    img = cv2.imread(imagePath)
    imgbox = pytesseract.image_to_boxes(img)
    imgH, imgW, _ = img.shape

    for box in imgbox.splitlines():
        box = box.split(' ')
        x1 = int(box[1])
        y1 = int(box[2])
        x2 = int(box[3])
        y2 = int(box[4])
        ch = box[0]
        if(ch.isalnum()):
            print(ch)
            cv2.rectangle(img, (x1,imgH - y1), (x2, imgH - y2), (255,0,0), 3)
    
    plt.imshow(img)
    plt.show()

def videoTextDetection(webcamon = True, videoPath="../assets/videos/Inglourious_Basterds_Intro.mp4"):
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX

    if(webcamon):
        videocap = cv2.VideoCapture(0)
        videocap.set(3,1280)
        videocap.set(4,720)
        videocap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        videocap = cv2.VideoCapture(videoPath)

    if (not videocap.isOpened()):
        cv2.VideoCapture(0)
    if (not videocap.isOpened()):
        raise IOError ("Cannot open video")

    counter = 0
    while True:
        ret,frame = videocap.read()
        counter += 1
        if((counter%3) == 0):

            imgH, imgW,_ =frame.shape
            x0,y0,w0,h0, = 0,0,imgH,imgW
            imagestring = pytesseract.image_to_string(frame)
            imgstr = re.sub(r'W+','',imagestring)
            imageboxes = pytesseract.image_to_boxes(frame)

            for box in imageboxes.splitlines():
                box = box.split(' ')
                x1,y1,w1,h1 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                ch = box[0]
                if(ch.isalnum()):
                    cv2.rectangle(frame, (x1, imgH- y1), (w1, imgH - h1), (255,0,0), 3)

            cv2.putText(frame, imagestring, ((x0 + int(w0/25)), (y0 + int(h0/25))), font, 1, (0,0,255), 2)

            cv2.imshow('balls',frame)
            plt.show

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
            
    videocap.release()
    cv2.destroyAllWindows()

n = len(sys.argv)

if(n == 1):
    print("no argument passed, exiting")
else:
    if(sys.argv[1] == "image"):
        if(n == 3):
            imageTextDetection(sys.argv[2])
        else:
            imageTextDetection()
    elif(sys.argv[1] == "video"):
        if(n==4):
            videoTextDetection(sys.argv[2], sys.argv[3])
        else:
            videoTextDetection()
    else:
        print("incorrect arguments")