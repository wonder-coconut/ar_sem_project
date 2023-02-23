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
3. (Subject to change) Navigate to text_recognition/ and run the test.py script
> cd text_recognition/  
> python3 test.py

As of this moment, the script takes in an image/video input from a local assets/(images/videos) folder. I have consciously git ignored the videos folder to save storage space, so you are free to plug in your own image/video and modify the script to run the media. Just uncomment the specific function call and add in your address path as a parameter.  
Future versions will void this in favour of command line arguments.

## Test output :   
### Image text detection :  
![Figure_1](https://user-images.githubusercontent.com/53872723/220915066-50005c26-6810-4deb-a3f1-9c5b8fd1d388.png)

### Video text detection :

## TODO
1. Improve video latency (currently skipping every 20th frame to improve performance)
2. Explore more accurate text recognition engines
3. Switch to live camera feed instead of a local video feed
