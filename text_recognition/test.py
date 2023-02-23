import pytesseract
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

img = cv2.imread('../assets/images/image1.jpg')
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
        print(f"{ch} - {ch.isalnum()}")
        cv2.rectangle(img, (x1,imgH - y1), (x2, imgH - y2), (255,0,0), 3)

