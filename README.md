# Text Detection and Recognition

Rishab Ramanathan   
19XJ1A0558

A work in progress build to create and develop text-based augmented reality, with the goal of school textbook applications in mind. Ar this stage, the application can run text detection on a video stream.

### System pre-requisites:
- python3-tk (Tkinter - for GUI representation of matplotlib, used for the purpose of viewing the output dynamically instead of writing to a file)
- tesseract-ocr (OCR library for character recognition. Use the following reference for installation : https://pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/)
- pip3 (Python package manager)

### Python pre-requisites:
- pytesseract
- opencv-contrib-python
- imutils
- matplotlib

## Installation:
1. (Optional) Set up the virtual environment using venv.
> python3 -m venv <em>/path/to/virtualenv/target/directory</em>  
> source <em>/path/to/virtualenv/target/directory/</em>bin/activate
2. Install the necessary library dependencies
> pip3 install -r requirements.txt

The directory contains 2 subparts : text recognition and augmented reality. The plan is to combine them both into a single project for text based anchoring for AR.

## Text Recognition :
At this stage, the script can detect and recognize text in real time in a webcam video stream. Two scripts have been made, one for purely text detection and the other for recognition of text. Text detection works at a much faster rate than recognition (naturally) but merely detecting the position of text does not prove useful for this project.

## Augmented Reality :
The current script can augment 3D models onto an augmented image using feature detection and homography. Further development has also enabled multiple simultaneous models augmented on different and unique augmented images.  
But as far as the goal is concerned, we need to augment the model onto a text based anchor/have some trigger based mechanism for rendering, based on text.
Issues:  
1. Text as a standalone cannot be used as an anchor point at this stage with this model of augmentation, as text contains too few feature points for unique detection.
2. One potential solution that was planned was to use the bounding boxes born out of text recognition, and use those coordinates to augment the model. Unfortunately, this won't be as possible as initially planned as the bounding box are coordinates of the screen's 2D plane. This same issue is resolved in AR using homography with the image target.
3. The final potentially succesful solution is to have some form of augmented image in each page of the textbook (perhaps like a school logo, publisher logo, etc), and use the text detected on the page to individually augment the specific model, depending on the text. Basically, use the augmented image as the anchor and the trigger for AR, and use the text to filter the model. 
Note : Image recognition accuracy (number of matches) improves by scaling down the image target. Need to explore. 
## Test output :   
### Image text detection :  
![Figure_1](https://user-images.githubusercontent.com/53872723/220915066-50005c26-6810-4deb-a3f1-9c5b8fd1d388.png)

### Video text detection :
https://user-images.githubusercontent.com/53872723/220920836-36a49481-18a4-4101-90f4-57a6f3f102f3.mp4


### Augmented Image feature point and bounding box :
https://user-images.githubusercontent.com/53872723/222406217-047c004f-0154-4507-a28a-bdf723ce80a6.mp4

### Video Calibration using checkerboard :
https://user-images.githubusercontent.com/53872723/227185833-1dd24517-bd4b-43b0-b735-36a087138660.mp4

### Augmented Image rendering :
https://user-images.githubusercontent.com/53872723/227186004-aa7d909f-9d97-4512-a220-6d224f57a3e3.mp4

### Multi Augmented Image rendering :
https://user-images.githubusercontent.com/53872723/227186093-15cbc0b9-e416-48b1-a263-e924fca2f300.mp4
