# Facial Detection and Recognition System for Video Analysis

## Project Overview
This project provides a GUI for face recognition using DeepFace, with options for single video and multi-video processing.
It also features a user-friendly graphical interface 

## Installation

### Prerequisites
- Python 3.9.12
- Git

#### Installing Python 3.9.12

1. For Windows:
   - Download the Python 3.9.12 installer from the official Python website: https://www.python.org/downloads/release/python-3912/
   - Run the installer and follow the installation wizard. Make sure to check "Add Python 3.9 to PATH" during installation.

2. For macOS:
   
   Option A: Using the official Python installer
   - Download the Python 3.9.12 installer for macOS from the official Python website: https://www.python.org/downloads/release/python-3912/
   - Open the downloaded .pkg file and follow the installation wizard.
   - After installation, open a new terminal and run:
     ```
     python3 --version
     ```
     This should output `Python 3.9.12`

   Option B: Using Homebrew (recommended for developers)
   - If you don't have Homebrew installed, install it first:
     ```
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Then install Python 3.9.12:
     ```
     brew install pyenv
     pyenv install 3.9.12
     pyenv global 3.9.12
     ```
   - Add the following to your shell configuration file (.zshrc, .bash_profile, or .bashrc):
     ```
     export PATH="$HOME/.pyenv/shims:$PATH"
     ```
   - Restart your terminal or run `source ~/.zshrc` (or your appropriate shell config file)
   - Verify the installation:
     ```
     python --version
     ```
     This should output `Python 3.9.12`

3. For Linux (Ubuntu/Debian):
   ```
   sudo apt update
   sudo apt install software-properties-common
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.9.12
   ```

   If `python3.9.12` is not available, you may need to install `python3.9` and then upgrade it:
   ```
   sudo apt install python3.9
   sudo apt install python3.9-distutils
   wget https://bootstrap.pypa.io/get-pip.py
   python3.9 get-pip.py
   python3.9 -m pip install --upgrade pip
   ```

   Verify the installation:
   ```
   python3.9 --version
   ```
   This should output `Python 3.9.12`

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/AKhromin/Computer-Vision-for-NGO-Problems.git
   cd Computer-Vision-for-NGO-Problems
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

There are 2 available modes for you to run the face recognition

1. Single Video Analysis:
   ```
   python run_face_recognition.py 1
   ```

2. Multi-Video Analysis:
   ```
   python run_face_recognition.py 2
   ```

## Key Features

- Single video processing
- Multi-video processing
- Known face detection
- Face detection with minimum size filter
- CSV logging of detected faces
- Summary report and graph generation

## Technical Specifications

For detailed technical requirements, please refer to `requirements.txt`.

## Customization

### Face Detection Model

To modify the face detection model, locate the following configuration in the source code:

```python
self.face_detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)
```
