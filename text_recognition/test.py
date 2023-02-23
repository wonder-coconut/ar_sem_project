import pytesseract
import cv2

img = cv2.imread('../assets/images/image2.jpg')
print(pytesseract.image_to_string(img))