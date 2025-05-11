# Person Camera Time Analysis Using Face Detection and Recognition

## Project Overview
This project provides a GUI for face recognition using DeepFace, for analyzing "camera time" of people shown on video.
The output of the system is a list of faces seen on video and how long each face was shown for.
Faces are saved as images, statistics as a CSV file, and a summary as a PDF.

## Installation (Windows)

- Download the Python 3.9.12 installer from the official Python website: https://www.python.org/downloads/release/python-3912/
- Run the installer and follow the installation wizard. Make sure to check "Add Python 3.9 to PATH" during installation.
- Install virtualenv:
  ```
  python -m pip install virtualenv
  ```
- Create a virtualenv environment and activate it.
- Install requirements:
  ```
  pip install tensorflow==2.10.0 tensorflow_cpu==2.10.0 tensorflow_intel==2.10.0 keras==2.10.0 deepface==0.0.93 numpy==1.26.4 matplotlib scipy torch pyinstaller
  ```

## Installation (Ubuntu 22.04, with virtualenv)

Tested on CPU, Python 3.10.12:

```
python -m pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install tensorflow==2.10.0 keras==2.10.0 deepface==0.0.93 numpy==1.26.4 matplotlib scipy torch pyinstaller
```

## Installation (Ubuntu 22.04, with venv, based on https://www.tensorflow.org/install/pip)

Tested on GPU, Python 3.10.12:

```
python3 -m venv tf
source tf/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda]
pip install deepface matplotlib scipy tf-keras torch
```

## Running

```
python GUI_single_vid.py
```

## Exporting as an executable (Windows)

- Run the following:
  ```
  pyinstaller --hiddenimport=matplotlib.backends.backend_pdf --collect-all cv2 GUI_single_vid.py
  ```
- Copy the model file (face_detection_yunet_2023mar.onnx) to same folder as the .exe file

