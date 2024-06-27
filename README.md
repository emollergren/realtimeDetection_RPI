# Raspberry-Object-Detection
Raspberry Pi object detection refers to the use of Raspberry Pi, a popular and affordable single-board computer, for executing object detection tasks. Object detection involves identifying and locating objects within an image or video feed, and it can be used for a variety of applications including security, automation, robotics, and more.

#### Why Use Raspberry Pi for Object Detection:
Affordability: The Raspberry Pi offers a cost-effective platform for hobbyists, educators, and developers to experiment with computer vision.
Portability: Its small form factor makes it ideal for deploying in mobile and embedded systems.
Community and Support: A large and active community provides resources, tutorials, and support for a wide array of projects.
### Key Components:
Raspberry Pi Board:
Models like Raspberry Pi 4 and Raspberry Pi 3 have sufficient processing power and memory to handle basic object detection tasks.
Camera Module:
Raspberry Pi Camera Module or USB webcams can be used to capture images and video.
Software Libraries:
OpenCV: An open-source computer vision library that provides tools for image and video analysis.
TensorFlow Lite: A lightweight version of TensorFlow optimized for mobile and embedded devices.
PyTorch: With its LibTorch library, it can also be used for deploying models on Raspberry Pi.
## Steps to Set Up Object Detection on Raspberry Pi:
Hardware Setup:

Connect the camera module or USB webcam to the Raspberry Pi.
Ensure proper power supply and connectivity (e.g., WiFi for remote access).
Software Installation:

Install Raspbian (Raspberry Pi OS) on the microSD card and boot up the Raspberry Pi.
Update the system and install required libraries:
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-pip
pip3 install opencv-python numpy
Install Deep Learning Frameworks:

For TensorFlow Lite:
pip3 install tflite-runtime
For PyTorch:
pip3 install torch torchvision
Load Object Detection Model:

Download or train a deep learning model suitable for your application, such as MobileNet-SSD or YOLO (You Only Look Once).
Pre-trained models designed specifically for edge devices are advantageous due to their optimized size and performance.
Write Detection Script:

Develop a Python script that utilizes the chosen library and model to perform object detection.
Example using OpenCV and TensorFlow Lite with a MobileNet-SSD model:
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
        
    # Preprocess the image to match model input requirements
    input_image = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Perform detection
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process and display results
    # (Implement your object detection result parsing and drawing here)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
Applications:
Security Systems: Monitor and detect unauthorized access or suspicious behavior.
Home Automation: Identify and interact with different objects or people for smart home systems.
Educational Projects: Provide hands-on experience in AI and computer vision.
Robotics: Enable robots to recognize and navigate around objects.
Challenges and Considerations:
Performance: Raspberry Pi has limited computational power compared to full-sized computers. Optimized models and efficient coding practices are essential.
Latency: Real-time processing may require careful management of resources.
Model Size: Use lightweight models to ensure they fit into the memory constraints of the Raspberry Pi.
Raspberry Pi object detection opens up exciting possibilities for innovative applications where cost, space, and power efficiency are crucial. It serves as a stepping stone for developers and hobbyists to explore the world of computer vision and artificial intelligence.
---
## Guidelines
* This repository contains python script for the object detection on Raspberry Pi in real time using OpenCV. 
* It uses a already trained [MobileNet](https://arxiv.org/abs/1704.04861) Architecture stored as Caffe Model. 
* This model uses Single Shot Detection([SSD](https://arxiv.org/abs/1512.02325)) algorithm for prediction.
* Look for the architecture detail [here](https://github.com/GopiKishan14/Raspberry-Object-Detection/blob/master/MobileNetSSD_deploy.prototxt.txt)
* This [code](https://github.com/GopiKishan14/Raspberry-Object-Detection/blob/master/pi_object_detection.py) stores the input images in a queue and output the predictions along with box in queue.
* This [code](https://github.com/GopiKishan14/Raspberry-Object-Detection/blob/master/real_time_object_detection.py) runs on real time videostream to output the prediction as well as bounding box across it.
* Above codes can be directly run on raspberry pi having packages [imutils](https://pypi.org/project/imutils/) and [cv2](https://pypi.org/project/opencv-python/)
* To Install, run
```
pip install imutils
pip install opencv-python
```
* USAGE, To run the videostream, clone the repo and in working directory, run
```
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```
Note : Press 'q' to stop the videostream.

---

## Abstract Code WalkThrough :
### Overviewing real_time_object_detection.py

* Importing packages

![image](https://user-images.githubusercontent.com/32811229/80908634-07b54e00-8d3f-11ea-9ddb-b553f04ba80a.png)

* Creating the argument parser for the python script
![image](https://user-images.githubusercontent.com/32811229/80908719-cf623f80-8d3f-11ea-9ce2-c21dd53b5e71.png)

* Initializing the list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box colors for each class.

![image](https://user-images.githubusercontent.com/32811229/80908757-2d8f2280-8d40-11ea-92a5-40403ec84015.png)

* Load the model and initialize the video stream.

![image](https://user-images.githubusercontent.com/32811229/80908774-57e0e000-8d40-11ea-916e-96373273c93b.png)

* For each frame, obtain the detections and predictions.

![image](https://user-images.githubusercontent.com/32811229/80908793-7ba42600-8d40-11ea-85d8-1f4aab2304de.png)


* For each detection, if prediction is above confidence (self-defined), draw the bounding box over it.

![image](https://user-images.githubusercontent.com/32811229/80908812-9d051200-8d40-11ea-897d-69385cca2291.png)

* Shows the output frame with break condition key "q". On Break condition, it destroys all the windows.

![image](https://user-images.githubusercontent.com/32811229/80908821-bf972b00-8d40-11ea-8733-a00f8244de93.png)

---

## Learning Resources
### For Learning basics of Raspberry PI 
[![Tutorial](http://img.youtube.com/vi/RpseX2ylEuw/0.jpg)](https://www.youtube.com/playlist?list=PLQVvvaa0QuDesV8WWHLLXW_avmTzHmJLv "Visit sentdex")

### For Learning basics of OpenCV

[![image](https://user-images.githubusercontent.com/32811229/80938999-b5d0fe80-8df8-11ea-95ad-3f6bae362ea4.png)](https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/)

---

## Fun Project for Contributors

### Face Detection using Haar Cascades

#### Refer to these tutorials for details.

[![image](https://user-images.githubusercontent.com/32811229/80939329-f67d4780-8df9-11ea-975a-b75659ec5470.png)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)


#### Refer to this cool video of sentdex
[![Tutorial](http://img.youtube.com/vi/88HdqNDQsEk/0.jpg)](https://www.youtube.com/watch?v=88HdqNDQsEk "Visit sentdex")
