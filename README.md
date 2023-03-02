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

## Test output :   
### Image text detection :  
![Figure_1](https://user-images.githubusercontent.com/53872723/220915066-50005c26-6810-4deb-a3f1-9c5b8fd1d388.png)


### Video text detection :
https://user-images.githubusercontent.com/53872723/220920836-36a49481-18a4-4101-90f4-57a6f3f102f3.mp4


### Augmented Image feature point and bounding box :
https://user-images.githubusercontent.com/53872723/222406217-047c004f-0154-4507-a28a-bdf723ce80a6.mp4


## TODO
1. Improve video latency (currently skipping every 20th frame to improve performance)
2. Explore more accurate text recognition engines
3. Switch to live camera feed instead of a local video feed
