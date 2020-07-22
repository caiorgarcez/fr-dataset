# import necessary packages
import numpy as np
import argparse
import cv2
import imutils
import time
import os
import dlib

# threads for faster fps
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream
from imutils import face_utils

# argument parser for imagesource and dnn libs
# USAGE: python main.py -isrc 0 -sol dlib -c 0 -o output -n Name1
ap = argparse.ArgumentParser()
ap.add_argument("-isrc", "--source", required = True, help = "Path of the video file.\n If the video source comes from a webcam set: 0.")
ap.add_argument("-sol", "--solution", required = True, help = "Solution for facial detection.\n Options: <dlib> or <resnet>")
ap.add_argument("-c", "--crop", required = True, help = "Crop output frame.\n Options: <0> for no <1> for yes.")
ap.add_argument("-o", "--output", required = True, help = "Path of the output folder for writing frames.")
ap.add_argument("-n", "--name", required = True, help = "Name of the person.")
ap.add_argument("-res", "--detector", type=str, default = "opencvresnet", help = "Folder of RESNET facial detector files compatible with the .dnn Opencv lib")
ap.add_argument("-conf", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# create a folder to hold captures
opath = os.path.sep.join([args["output"], args["name"]])
os.mkdir(opath)

# count variable to enumerate the frames
count = 0

# load files from the RESNET-X detector or dlib's HOG-based frontal face detector
if args["solution"] == "resnet":
    prototxtPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    res10net = cv2.dnn.readNet(prototxtPath, weightsPath)
elif args["solution"] == 'dlib':
    dlib_detector = dlib.get_frontal_face_detector()

# initialize threads for capturing frames from source file
if args["source"] == "0": 
    vs = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
else:
    fvs = FileVideoStream(args["source"]).start()
    time.sleep(1.0)

# face detection function for resnet
def resnet_detection(frame, res10net):
    # construct a blob from the image and pass trought the previously loaded net
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    res10net.setInput(blob)
    detections = res10net.forward()

    # loop over detections
    for i in range(0, detections.shape[2]):
        
        # extract confidence prob. of eache 
        confidence = detections[0, 0, i, 2]
        
        # select ROI and return cropped frame case needed
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and resize it to 224x224
            if args["crop"] == "1":
                crop = frame[startY:endY, startX:endX]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(gray, (224, 224))
            elif args["crop"] == "0":
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return face
    
# face detection function for dlib
def dlib_detection(frame, dlib_detector):
    # detect faces in the grayscale image without upsampling
    rects = dlib_detector(frame, 0) 

    # loop over detections
    for (i, rect) in enumerate(rects):
        # convert dlib's rectangle to a OpenCV-style bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect) 

        # select ROI and return cropped frame case needed
        if args["crop"] == "1":
            crop = frame[y:(y+h), x:(x+w)]
            face = cv2.resize(crop, (224, 224))
        elif args["crop"] == "0":
            face = frame
        return face

# process threaded video stream and output cropped frame
while True:
    # opencv commands for keyborad
    key = cv2.waitKey(1) & 0xFF

    if args["source"] == "0":
	    frame = vs.read()
    else:   
        frame = fvs.read()
    
    # resize the frame to 300x300 pixels
    frame = imutils.resize(frame, width=300)

    # call functions for face detection
    if args["solution"] == "resnet":
        face = resnet_detection(frame, res10net)
    # dlib is set to work with grayscaled frames
    elif args["solution"] == 'dlib':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = dlib_detection(gray, dlib_detector)

        if face is not None:
            cv2.imshow("face", face)
            # press k to write a frame to output folder
            if key == ord("k"):
                fpath = os.path.sep.join([opath, f'{args["name"]}{str(count).zfill(5)}.png'])
                cv2.imwrite(fpath, face)
                count += 1
        else:
            print("No detection")   

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()

# Stop videostream threads
if args["source"] == "0": 
    vs.stop()
else:
    fvs.stop()
