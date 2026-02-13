# Class-Sleep-Monitor
# Student Sleep Tracker

**Project Description:**  
This project is a Python-based system that monitors students in a classroom using a webcam and detects if they are sleeping. The system uses **MediaPipe Face Mesh** to track facial landmarks and calculates the **Eye Aspect Ratio (EAR)** to determine if the eyes are closed. Sleeping students’ faces are displayed in a separate window for easy monitoring by teachers.

---

## Features

- Real-time face tracking for up to 20 students.
- Detects closed eyes using Eye Aspect Ratio (EAR).
- Highlights sleeping students in a separate window.
- Displays the count of sleeping students in a **Tkinter GUI**.
- Works with standard webcams.

---

## Requirements

- Python 3.8–3.9
- OpenCV (`pip install opencv-python`)
- MediaPipe 0.8.10 (`pip install mediapipe-0.8.10-cp38-cp38-win_amd64.whl`)
- NumPy (`pip install numpy`)
- Tkinter (usually comes with Python)

**Note:** MediaPipe version 0.8.10 is recommended for Python 3.8. Using newer versions may cause errors.

---

## Installation

1. Install Python 3.8–3.9.
2. Install required packages:

```bash
pip install numpy opencv-python
pip install path/to/mediapipe-0.8.10-cp38-cp38-win_amd64.whl
