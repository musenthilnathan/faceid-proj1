# FaceID-Proj1

A simple, user-friendly face recognition and face extraction toolkit for non-programmers. Works on Windows/Linux/Mac with Python 3.8+.

## Features
- Add faces to a database using your webcam or image files
- Identify people in new photos using the database
- Extract all faces from a group photo and save them as separate images
- No coding required for basic use

## Folder Structure
```
faceid-proj1/
├── identifywithcamera.py   # Main script for face database and recognition
├── extractFaces.py         # Script to extract faces from group photos
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

## Setup Instructions

### 1. Install Python
- Download and install Python 3.8 or newer from [python.org](https://www.python.org/downloads/).
- Make sure to check "Add Python to PATH" during installation.

### 2. Download the Project
- Click the green "Code" button on GitHub, then "Download ZIP".
- Unzip the folder to your computer.
- Or, use `git clone` if you are familiar with git.

### 3. Open a Terminal/Command Prompt
- Navigate to the `faceid-proj1` folder:
  - On Windows: `cd path\to\faceid-proj1`
  - On Mac/Linux: `cd /path/to/faceid-proj1`

### 4. Create a Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 5. Install Dependencies
```
pip install -r requirements.txt
```

### 6. Install System Packages (Linux only)
- For camera window always-on-top: `sudo apt-get install wmctrl`
- For HEIC image support: `sudo apt-get install imagemagick libheif1`

## Usage

### A. Face Database & Recognition
Run:
```
python identifywithcamera.py
```
- Follow the prompts to add faces (from file or camera).
- After adding, you can test recognition with new images.
- The script will tell you if a match is found or not.

### B. Extract Faces from Group Photo
Run:
```
python extractFaces.py
```
- Enter the path to your group photo (JPG/PNG/HEIC).
- The script will extract all faces and let you save them one by one.

## Notes
- All data is stored locally in `faces.db`.
- For best results, add several images per person (with/without glasses, different lighting, etc).
- No internet connection required after install.

## Troubleshooting
- If you get errors about missing packages, re-run `pip install -r requirements.txt`.
- If the camera window is hidden, check your taskbar or minimize other windows.
- For HEIC images, make sure ImageMagick is installed and working.

## License
MIT License. Free for personal and educational use.
