## fr-dataset
Script for generating a facial recognition dataset for deep learning applications. 

-----

### 1. Install the environment

Packages and other requirements to run `fr-dataset` is provided for conda in `requirements.yml` and for pip in `requirements.txt`.

for conda run: 
`$ conda env create -f requirements.yml`

for pip run:
`$ pip install -r requirements.txt` 

-----

### 2. Usage and features:

`ap = argparse.ArgumentParser()

ap.add_argument("-isrc", "--source", required = True, help = "Path of the video file.\n If the video source comes from a webcam set: 0.")
ap.add_argument("-sol", "--solution", required = True, help = "Solution for facial detection.\n Options: <dlib> or <resnet>")
ap.add_argument("-c", "--crop", required = True, help = "Crop output frame.\n Options: <0> for no <1> for yes.")
ap.add_argument("-o", "--output", required = True, help = "Path of the output folder for writing frames.")
ap.add_argument("-n", "--name", required = True, help = "Name of the person.")
ap.add_argument("-res", "--detector", type=str, default = "opencvresnet", help = "Folder of RESNET facial detector files compatible with the .dnn Opencv lib")
ap.add_argument("-conf", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")`



----- 

### 3. Observations:

Within the project repository it is necessary to create a folder named `output` to store the captured frames. Later, provide in the parse the name of the folder as in:
`$ ... -o <output path>` 

The code for this project is based in: [[1]](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
