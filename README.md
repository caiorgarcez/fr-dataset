## fr-dataset
Script for generating a custom facial recognition dataset for deep learning applications. 

-----

### 1. Install the environment

Packages and other requirements to run `fr-dataset` is provided for conda in `requirements.yml` and for pip in `requirements.txt`.

for conda run: 
`$ conda env create -f requirements.yml`

for pip run:
`$ pip install -r requirements.txt` 

-----

### 2. Usage and features:

`-isrc` Path of the video file. If the video source comes from a webcam set: <0>  
`-sol` Solution for facial detection. Options: <dlib> or <resnet>  
`-c` Crop output frame.Options: <0> for no <1> for yes  
`-o` Path of the output folder for writing frames  
`-n` Name of the person
`-res` Folder of RESNET facial detector files compatible with the .dnn Opencv lib  
`-conf` Minimum probability to filter weak detections  

Example: `$ python main.py -isrc 0 -sol dlib -c 0 -o output -n <name1>` 

----- 

### 3. Observations:

Within the project repository it is necessary to create a folder named `output` to store the captured frames. Later, provide in the parse the name of the folder as in:
`$ ... -o <output path>` 

The code for this project is based in: [[1]](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
