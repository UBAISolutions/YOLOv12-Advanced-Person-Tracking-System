# YOLOv12 Person Tracking GUI Application

## Overview
An advanced person tracking system with a user-friendly graphical interface built using Python, YOLOv12, OpenCV, and Tkinter. This application provides real-time person detection, face recognition for re-identification, and comprehensive reporting capabilities.

## Features

### 🎯 Core Functionality
- **YOLOv12 Object Detection**: State-of-the-art person detection with high accuracy
- **Face Recognition**: OpenCV-based face recognition to prevent duplicate counting
- **Person Re-identification**: Tracks individuals even when they leave and re-enter the frame
- **Real-time Processing**: Live tracking with visual feedback and statistics

### 📱 User Interface
- **Tabbed Interface**: Clean, organized GUI with separate tabs for different functions
- **Multiple Input Sources**: Support for webcam, external cameras, and video files
- **Real-time Statistics**: Live display of tracking metrics and counts
- **Visual Feedback**: Bounding boxes, person IDs, and tracking trails

### 📊 Reporting System
- **Excel Reports**: Comprehensive tracking reports in Excel format
- **Session Analytics**: Detailed statistics about each tracking session
- **People Summary**: Individual person tracking data
- **Report Viewer**: Built-in report viewing and folder access

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Required Packages
The following packages are automatically installed:
```
ultralytics>=8.3.0
opencv-python>=4.5.0
pandas>=1.3.0
openpyxl>=3.0.0
numpy>=1.21.0
scipy>=1.7.0
Pillow>=8.0.0
```

### Setup Instructions
1. **Clone or download** the project files
2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   ```
3. **Activate virtual environment**:
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python pandas openpyxl numpy scipy Pillow
   ```

## Usage Guide

### Starting the Application
```bash
python person_tracking_gui.py
```

### Main Interface

#### 📹 Webcam Tab
- **Purpose**: Track people using your computer's built-in camera
- **Controls**:
  - "▶ Start Webcam": Begin webcam tracking
  - "⏹ Stop Tracking": Stop tracking and generate report
- **Instructions**:
  1. Click "Start Webcam" to begin tracking
  2. Position yourself in front of the camera
  3. Move around to test re-identification
  4. Press ESC key or "Stop" button to end session

#### 📷 External Camera Tab
- **Purpose**: Use external USB cameras or secondary cameras
- **Controls**:
  - Camera Index dropdown (0, 1, 2, 3)
  - "▶ Start External Camera": Begin external camera tracking
  - "⏹ Stop Tracking": Stop tracking and generate report
- **Instructions**:
  1. Select appropriate camera index (try 0, 1, 2, etc.)
  2. Click "Start External Camera"
  3. Ensure camera is properly connected
  4. Press ESC key or "Stop" button to end session

#### 🎬 Video File Tab
- **Purpose**: Analyze pre-recorded video files
- **Controls**:
  - "📁 Browse": Select video file
  - "▶ Start Video Analysis": Begin video analysis
  - "⏹ Stop Analysis": Stop analysis and generate report
- **Supported Formats**: MP4, AVI, MOV, MKV, WMV, FLV
- **Instructions**:
  1. Click "Browse" to select video file
  2. Click "Start Video Analysis"
  3. Video will play with real-time tracking overlay
  4. Press ESC key or "Stop" button to end analysis

#### 📊 Reports Tab
- **Purpose**: View and manage tracking reports
- **Controls**:
  - "📈 View Latest Report": Display most recent tracking report
  - "📁 Open Reports Folder": Open folder containing all reports
- **Features**:
  - Session statistics and analytics
  - Individual person tracking data
  - Excel file generation with timestamps

### Tracking Controls
- **ESC Key**: Stop tracking from any mode (primary method)
- **Stop Buttons**: Alternative way to stop tracking from GUI
- **Real-time Display**: Live statistics and person counts
- **Visual Indicators**: Bounding boxes, person IDs, tracking trails

## Technical Details

### Person Detection
- **Model**: YOLOv12 Nano (yolo12n.pt)
- **Confidence Threshold**: 0.5 (50% minimum confidence)
- **Input Resolution**: 640x480 pixels for optimal performance
- **Processing Speed**: ~30 FPS on modern hardware

### Face Recognition
- **Method**: OpenCV Haar Cascade + Histogram Analysis
- **Features**: 256-dimensional histogram vectors
- **Similarity Metric**: Correlation coefficient
- **Threshold**: 0.6 (60% similarity for same person)
- **Memory Management**: Automatic cleanup of old person data

### Tracking Algorithm
- **Re-identification**: Face-based person matching
- **Trail History**: 30-frame tracking trail visualization
- **Timeout**: 15 seconds (450 frames) before forgetting a person
- **Maximum People**: 30 simultaneous tracked individuals

## Output Files

### Video Output
- **Location**: `data/tracking_output_YYYYMMDD_HHMMSS.mp4`
- **Format**: MP4 with H.264 encoding
- **Content**: Original video with tracking overlays
- **Resolution**: 640x480 pixels

### Excel Reports
- **Location**: `data/tracking_report_YYYYMMDD_HHMMSS.xlsx`
- **Sheets**:
  - **Session_Info**: Overall session statistics
  - **People_Summary**: Individual person data
- **Contents**:
  - Session duration and timing
  - Unique people count
  - Total detections
  - Person-specific tracking data
  - Technical configuration details

## Troubleshooting

### Common Issues

#### Camera Not Working
- **Solution**: Try different camera indices (0, 1, 2, 3)
- **Check**: Ensure camera is not being used by another application
- **Verify**: Camera permissions are granted to Python

#### Video File Not Loading
- **Solution**: Ensure file format is supported (MP4 recommended)
- **Check**: File path contains no special characters
- **Verify**: Video file is not corrupted

#### Poor Face Recognition
- **Solution**: Ensure good lighting conditions
- **Check**: Face is clearly visible and not obscured
- **Adjust**: Stay within 1-3 meters of camera for optimal recognition

#### Performance Issues
- **Solution**: Close other applications using camera/CPU
- **Check**: Available system memory and CPU usage
- **Optimize**: Use webcam tab for best performance

### Error Messages

#### "Could not open video source"
- **Cause**: Camera unavailable or video file corrupted
- **Solution**: Check camera connection or try different video file

#### "Failed to generate report"
- **Cause**: File permission issues or insufficient disk space
- **Solution**: Ensure write permissions to 'data' folder

## Advanced Configuration

### Modifying Parameters
Key parameters can be adjusted in the code:

```python
# Face recognition threshold (0.0-1.0, higher = more strict)
face_similarity_threshold = 0.6

# Person timeout (frames, 30 fps = 1 second)
person_timeout = 450  # 15 seconds

# Maximum tracked people
max_known_people = 30

# Confidence threshold for person detection
confidence_threshold = 0.5
```

### Custom Video Resolution
```python
# In start_tracking method
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Change width
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Change height
```

## System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD equivalent
- **RAM**: 8GB system memory
- **Storage**: 2GB free space
- **Camera**: USB 2.0 compatible webcam
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **CPU**: Intel i7 or AMD Ryzen 7
- **RAM**: 16GB system memory
- **GPU**: Dedicated graphics card (optional, for faster processing)
- **Storage**: 5GB free space
- **Camera**: USB 3.0 HD webcam

## License and Credits

### Model Credits
- **YOLOv12**: Ultralytics (https://github.com/ultralytics/ultralytics)
- **OpenCV**: Open Source Computer Vision Library
- **Face Detection**: Haar Cascade Classifiers

### Dependencies
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Pillow**: Image processing
- **Tkinter**: GUI framework (built-in with Python)

## Support and Updates

### Getting Help
1. Check this README for common solutions
2. Verify all dependencies are installed correctly
3. Ensure proper camera/file permissions
4. Test with different input sources

### Version Information
- **Current Version**: 1.0
- **Last Updated**: August 12, 2025
- **Python Compatibility**: 3.8+
- **YOLOv12 Model**: Nano (yolo12n.pt)

---

**Note**: This application is designed for educational and research purposes. Performance may vary based on system specifications and environmental conditions.
