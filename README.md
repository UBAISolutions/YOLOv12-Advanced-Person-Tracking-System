# 🎯 YOLOv12 Advanced Person Tracking System

A sophisticated real-time person tracking system using YOLOv12 with facial recognition-based re-identification, movement trail visualization, and comprehensive analytics reporting.

## 🌟 Features

### 🔍 **Advanced Detection & Tracking**
- **YOLOv12 Nano Model** for fast and accurate person detection
- **Facial Recognition** using OpenCV histogram analysis for person re-identification
- **Persistent Person IDs** that survive occlusions and temporary disappearances
- **False Positive Filtering** (5-frame confirmation before ID assignment)
- **Memory Management** with automatic cleanup of old person data

### 📊 **Real-time Analytics**
- **Movement Trails** showing the path of each tracked person (last 30 positions)
- **Live Statistics** with detection counts and progress tracking
- **Comprehensive Excel Reports** with detailed tracking data
- **Session Analytics** including duration, detection rates, and person statistics

### 🎥 **Video Processing**
- **Automatic FPS Detection** maintains original video timing
- **Full Resolution Output** preserves original video quality
- **Real-time Preview** with 75% scaled display window
- **Progress Monitoring** with frame-by-frame updates

### 📈 **Data Export**
- **Excel Reports** with multiple sheets (Session Info, People Summary, Detailed Tracking)
- **Person Statistics** including duration, distance traveled, and detection rates
- **Timestamp Tracking** for first/last appearance of each person

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+ 
- Webcam or video files in supported formats (MP4, AVI, MOV, MKV)

### 2. Installation
```bash
# Clone the repository
git clone <your-repository-url>
cd YOLOv12-Person-Tracking

# Install dependencies
pip install -r requirements.txt
```

### 3. Download YOLOv12 Model
The script will automatically download the YOLOv12 nano model (`yolo12n.pt`) on first run, or you can place it manually in the project directory.

### 4. Run the Tracker
```bash
# Set environment variable to avoid OpenMP conflicts (Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

# Run the tracking system
python yolo-12.py
```

### 5. Controls
- **Press 'q'** to quit the video analysis early
- **Close window** to stop processing
- **Check terminal** for real-time progress updates

## 📦 Dependencies

```txt
ultralytics>=8.3.0    # YOLOv12 model support
opencv-python>=4.8.0  # Computer vision operations  
numpy>=1.21.0         # Numerical computations
pandas>=1.3.0         # Excel report generation
scipy>=1.7.0          # Distance calculations
```

## 📁 Project Structure

```
├── yolo-12.py                    # Main tracking script
├── requirements.txt              # Python dependencies
├── yolo12n.pt                   # YOLOv12 Nano model weights (auto-downloaded)
├── data/                        # Video files directory
│   ├── Subway.mp4              # Default input video
│   └── video_analysis_yolo12.mp4 # Generated output video
├── tracking_report_*.xlsx       # Generated Excel reports
└── README.md                    # This file
```

## ⚙️ Configuration

### Default Settings
```python
# Video Configuration
video_path = "data/Subway.mp4"        # Input video file
output_path = "data/video_analysis_yolo12.mp4"  # Output video file

# Detection Settings
confidence_threshold = 0.25           # Minimum detection confidence
face_similarity_threshold = 0.6       # Face matching threshold (0-1)
min_face_size = 50                   # Minimum face size for recognition
trail_length = 30                    # Number of trail points to store

# Memory Management
max_known_people = 30                # Maximum stored face encodings
person_timeout = 450                 # Frames before forgetting a person (15s @ 30fps)
```

### Changing Input Video
To use a different video, modify line 23 in `yolo-12.py`:
```python
video_path = "data/your_video.mp4"
```

## 🎯 Key Features Explained

### 🔄 **Person Re-identification**
- Uses OpenCV face detection with Haar cascades
- Extracts facial histograms for unique person fingerprints
- Maintains consistent IDs even when people leave and re-enter the frame
- Correlation-based similarity matching with 0.6 threshold

### 🎨 **Visual Elements**
- **Bounding Boxes**: Blue rectangles around detected persons
- **Person IDs**: Large ID numbers displayed above each person
- **Movement Trails**: Blue lines connecting recent positions
- **Center Points**: Tracks center of each bounding box
- **Information Overlay**: Real-time statistics panel

### 📊 **Analytics Dashboard**
The overlay panel displays:
- Model information (YOLOv12 Nano + OpenCV Face Recognition)
- Video information and progress
- Frame counter and processing percentage
- Unique people count and active IDs
- Real-time detection statistics

## 🛠️ Technical Details

### **Detection Pipeline**
1. **YOLO Detection**: YOLOv12 detects persons with bounding boxes
2. **Face Extraction**: OpenCV extracts face regions from person boxes
3. **Feature Encoding**: Generates histogram-based face encodings
4. **Similarity Matching**: Compares with known faces using correlation
5. **ID Assignment**: Assigns persistent IDs based on facial recognition
6. **Trail Tracking**: Updates movement trails and statistics

### **Performance Optimizations**
- **Lazy Initialization**: Video writer initialized after first frame
- **Memory Cleanup**: Automatic removal of old person data
- **Efficient Trails**: Deque structure for O(1) trail updates
- **Filtered Detection**: 5-frame confirmation reduces false positives

### **Output Generation**
- **Full Resolution Video**: Maintains original video quality and FPS
- **Excel Reports**: Multi-sheet reports with comprehensive analytics
- **Real-time Monitoring**: Progress updates every 30 frames

## 📊 Performance Metrics

- **Processing Speed**: ~20-30 FPS on modern hardware
- **Memory Usage**: Automatic cleanup prevents memory bloat
- **Detection Accuracy**: High precision with false positive filtering
- **Re-identification**: Robust facial recognition-based matching

## 🔧 Troubleshooting

### **Common Issues**

**Video Won't Open**
```bash
Error: Could not open video file
```
- Verify video file exists in `data/` folder
- Check supported formats (MP4, AVI, MOV, MKV)
- Ensure video file is not corrupted

**OpenMP Error**
```bash
OMP: Error #15: Initializing libiomp5md.dll
```
Solution:
```bash
# Windows
set KMP_DUPLICATE_LIB_OK=TRUE

# Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
```

**No Face Detection**
- Ensure good lighting in video
- Check minimum face size settings
- Verify people are facing camera (at least partially)

**Slow Performance**
- Use smaller input videos
- Reduce trail length
- Lower face similarity threshold

## 📈 Output Files

### **Video Output**
- **File**: `data/video_analysis_yolo12.mp4`
- **Format**: MP4 with original resolution and FPS
- **Content**: Annotated video with bounding boxes, IDs, and trails

### **Excel Reports**
- **File**: `tracking_report_YYYYMMDD_HHMMSS.xlsx`
- **Sheets**:
  - **Session_Info**: Overall session statistics
  - **People_Summary**: Individual person tracking statistics  
  - **Detailed_Tracking**: Frame-by-frame tracking data

## 🎯 Use Cases

- **Security Monitoring**: Track people in surveillance footage
- **Retail Analytics**: Analyze customer movement patterns
- **Sports Analysis**: Track player movements and statistics
- **Research**: Study human behavior and movement patterns
- **Event Monitoring**: Count and track attendees

## 🔬 Technical Specifications

- **Model**: YOLOv12 Nano (yolo12n.pt)
- **Face Recognition**: OpenCV Haar Cascade + Histogram Analysis
- **Tracking**: Object persistence with facial re-identification
- **Memory**: Automatic cleanup with configurable limits
- **Export**: Pandas-based Excel report generation

## 🧠 How It Works

1. **Detection & Tracking**: YOLOv12 detects persons and assigns tracking IDs (`model.track` with `persist=True`)
2. **Custom ID Mapping**: Internal YOLO IDs are mapped to stable sequential IDs
3. **Face Recognition**: Extracts facial features and compares with known faces
4. **Trail Drawing**: A `deque` stores recent positions per ID; OpenCV draws connecting lines
5. **False Positive Filtering**: Only objects appearing for 5+ consecutive frames get stable IDs
6. **Analytics Generation**: Comprehensive statistics and Excel reports

## 📸 Sample Output

The system generates:
- Real-time video with bounding boxes, person IDs, and movement trails
- Comprehensive Excel reports with tracking analytics
- Progress monitoring with frame-by-frame updates

## 🎮 Example Usage

```python
# Basic usage
python yolo-12.py

# With environment variable (Windows)
set KMP_DUPLICATE_LIB_OK=TRUE && python yolo-12.py

# Check available videos
ls data/
```

## � License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

## 🏆 Acknowledgments

- YOLOv12 by Ultralytics for state-of-the-art object detection
- OpenCV for computer vision operations
- The open-source computer vision community

---

**Author**: [Your Name]  
**Repository**: [Your GitHub Repository URL]  
**Last Updated**: August 2025

- **YOLOv12 Nano Model** for fast and accurate person detection
- **Face Recognition** using OpenCV histogram analysis to prevent duplicate counting
- **Person Re-identification** to track individuals even when they leave and re-enter frame
- **Movement Trails** showing the path of each tracked person
- **Real-time Statistics** with detection counts and progress tracking
- **Clean Visual Display** with blue-themed annotations

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Tracker
```bash
python clean_tracking.py
```

### 3. Controls
- Press **'q'** to quit the video analysis
- Video window shows at 75% scale for better screen fit

## 📦 Dependencies

- **ultralytics** - YOLOv12 model support
- **opencv-python** - Computer vision operations
- **numpy** - Numerical computations

## 📁 Project Structure

```
├── clean_tracking.py      # Main tracking script
├── requirements.txt       # Python dependencies
├── yolo12n.pt            # YOLOv12 Nano model weights
├── data/                 # Video files directory
│   └── Subway2.mp4       # Sample video file
└── README.md             # This file
```

## ⚙️ Configuration

Default video path: `data/Subway2.mp4`

To use a different video, modify line 13 in `clean_tracking.py`:
```python
def __init__(self, video_path="data/your_video.mp4", similarity_threshold=0.6):
```

## 🎯 Key Features

### Detection Settings
- **Confidence Threshold**: 0.25 (25% minimum confidence)
- **Face Similarity**: 0.6 correlation threshold
- **Trail Length**: 30 recent positions stored
- **Display Size**: 75% of original video resolution

### Visual Elements
- **Blue Bounding Boxes**: `(255, 0, 0)` in BGR format
- **Person IDs**: Large blue font labels
- **Movement Trails**: Blue lines connecting recent positions
- **Center Points**: Blue filled circles at person centers

## 🛠️ Technical Details

- **Model**: YOLOv12 Nano (yolo12n.pt)
- **Face Recognition**: OpenCV Haar Cascade + Histogram Analysis
- **Memory Management**: Automatic cleanup of old person data
- **Video Backends**: FFMPEG with fallback to default

## 📊 Performance

- **Processing Speed**: ~30 FPS on modern hardware
- **Memory Usage**: Limited to 30 known faces maximum
- **Detection Range**: Works best at 1-5 meter distances

## � Troubleshooting

### No Detections Visible
- Check if video file exists in `data/` folder
- Ensure good lighting in video
- Try lowering confidence threshold in code

### Video Won't Open
- Verify video format is supported (MP4, AVI, MOV, MKV)
- Check file path is correct
- Ensure video file is not corrupted

## 📝 Usage Rights

This is a clean, simplified implementation for educational and personal use.

---

**Note**: This is a simplified, clean version focused on core functionality.

## 📂 Project Structure
```

.
├── data/people_walking.mp4             # Input video file
├── people_tracking.py                  # Main tracking script
├── yolov8n.pt                          # YOLOv8 Nano model weights
├── data/people_walking_output.mp4      # Output video (generated)
└── README.md

````

## 🚀 Installation

1. **Clone this repository**:
```bash
git clone https://github.com/di37/yolov8-person-tracker.git
cd yolov8-person-tracker
````

2. **Install dependencies**:

```bash
pip install ultralytics opencv-python
```

3. **Download YOLOv8 model weights**:

```bash
yolo download model=yolov8n.pt
```

Alternatively, you can manually place `yolov8n.pt` in the project directory.

## ▶️ Usage

Run the script with:

```bash
python people_tracking.py
```

* Press **`q`** to quit early.
* The processed video will be saved as `people_with_trail_output.mp4`.

## ⚙️ Key Parameters

* `classes=[0]`: Detect only persons.
* `maxlen=30`: Number of recent points stored for trails.
* `appear[oid] >= 5`: Minimum frames before confirming a person.
* `fps`: Dynamically detected from input video; defaults to 30.

---

**Author**: [Your Name]  
**Repository**: [Your GitHub Repository URL]  
**Last Updated**: August 2025
