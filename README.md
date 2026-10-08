# 🚘 Automatic Number Plate Recognition Using YOLOv8, EasyOCR and Fog Detection

## 📌 Overview

This project implements an intelligent **Automatic Number Plate Recognition (ANPR)** system using **YOLOv8, EasyOCR, OpenCV, PyTorch, and fog detection techniques**.

The system is designed to detect vehicle license plates from **images, videos, and live webcam streams**, extract the license plate region using YOLOv8, and recognize the characters present on the plate using EasyOCR.

The system also analyzes **foggy or low-visibility conditions** and performs image enhancement to improve the quality of the input before OCR processing.

The complete pipeline combines:

- Deep learning-based object detection
- Computer vision
- Optical Character Recognition
- Fog/visibility analysis
- Image enhancement
- Real-time video processing
- Automated result generation

---

# 🚀 Features

- License plate detection using **YOLOv8**
- License plate text recognition using **EasyOCR**
- Real-time webcam ANPR
- Image-based ANPR
- Video-based ANPR
- Fog/haze detection
- Image enhancement for low-visibility conditions
- OCR preprocessing
- License plate cropping
- YOLO detection confidence
- OCR confidence
- Indian license plate format validation
- Annotated output images
- CSV result generation
- CPU-based execution support
- Custom YOLOv8 license plate model training
- Validation of the trained detection model

---

# 🏗️ System Architecture

```text
Camera / Image / Video
          ↓
    Fog Detection
          ↓
   Image Enhancement
          ↓
       YOLOv8
          ↓
 License Plate Detection
          ↓
    Plate Cropping
          ↓
  OCR Preprocessing
          ↓
      EasyOCR
          ↓
License Plate Number
          ↓
 Annotated Image / CSV
```

---

# 🧠 Complete System Workflow

```text
Input Source
   │
   ├── Image
   ├── Video
   └── Webcam
        │
        ▼
Frame Acquisition
        │
        ▼
Fog / Visibility Analysis
        │
        ├── Clear
        │
        └── Foggy
              │
              ▼
       Image Enhancement
        │
        ▼
YOLOv8 License Plate Detection
        │
        ▼
License Plate Bounding Box
        │
        ▼
License Plate Crop
        │
        ▼
OCR Preprocessing
        │
        ├── Resize
        ├── Grayscale
        ├── CLAHE
        └── Noise Reduction
        │
        ▼
EasyOCR
        │
        ▼
Text Extraction
        │
        ▼
Character Filtering
        │
        ▼
License Plate Number
        │
        ▼
Final ANPR Result
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 | License plate detection |
| Ultralytics | YOLOv8 implementation |
| EasyOCR | License plate character recognition |
| OpenCV | Image and video processing |
| PyTorch | Deep learning framework |
| NumPy | Numerical computation |
| Pandas | CSV/result processing |
| Pillow | Image processing support |
| Git | Version control |
| GitHub | Source code hosting |

---

# 📂 Project Structure

```text
ANPR-YOLOv8-EasyOCR-Fog-Detection/
│
├── app.py
├── realtime_anpr.py
├── video_anpr.py
│
├── plate_detector.py
├── fog_detector.py
├── ocr_engine.py
├── utils.py
├── annotate.py
│
├── train_plate.py
├── split_dataset.py
├── plate_data.yaml
│
├── requirements.txt
├── README.md
├── DATASET_INSPECTION.txt
├── .gitignore
│
├── models/
│   └── plate_best.pt
│
├── data/
│   └── labels/
│
└── outputs/
    ├── anpr_results.csv
    └── annotated/
```

> Dataset images, generated outputs, training runs, and large model files may be excluded from GitHub using `.gitignore`.

---

# ⚙️ Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/Dhanushini-S21/ANPR-YOLOv8-EasyOCR-Fog-Detection.git
```

Move into the project directory:

```powershell
cd ANPR-YOLOv8-EasyOCR-Fog-Detection
```

---

# 2. Create a Virtual Environment

For Windows:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, PowerShell should display:

```text
(.venv)
```

before the terminal path.

---

# 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

Main dependencies include:

```text
ultralytics
easyocr
opencv-python
torch
torchvision
numpy
pandas
Pillow
```

---

# 📦 Model Setup

The trained YOLOv8 license plate detection model is:

```text
models/plate_best.pt
```

The model is used by the real-time ANPR application.

Place the model in:

```text
models/
└── plate_best.pt
```

If the trained model is not available, it can be recreated by training the YOLOv8 model using the project dataset.

---

# 📊 Dataset

The project uses vehicle images containing visible license plates.

The dataset preparation process consists of:

```text
Vehicle Images
      ↓
License Plate Annotation
      ↓
YOLO Format Labels
      ↓
Dataset Splitting
      ↓
Training Dataset
      ↓
Validation Dataset
      ↓
YOLOv8 Training
      ↓
Model Validation
```

---

# 📁 Dataset Structure

The dataset can be organized as:

```text
data/
│
├── images/
│
├── images 01/
│   └── video_images/
│       ├── image_001.jpg
│       ├── image_002.jpg
│       ├── image_003.jpg
│       └── ...
│
└── labels/
```

The main image dataset used for ANPR processing is:

```text
data/images 01/video_images
```

---

# 🏷️ License Plate Annotation

The license plates were annotated using the YOLO bounding-box format.

Each annotation follows:

```text
class_id center_x center_y width height
```

Example:

```text
0 0.488528 0.492404 0.163799 0.064343
```

The detection class is:

```text
0 → license_plate
```

---

# 📈 Dataset Split

The YOLO training dataset was divided into:

```text
Training Images   : 153
Validation Images : 39
Total Annotated   : 192
```

The dataset contains one detection class:

```text
0 → license_plate
```

---

# 📄 YOLO Dataset Configuration

The dataset configuration is stored in:

```text
plate_data.yaml
```

Configuration:

```yaml
path: data/plate_dataset
train: images/train
val: images/val

names:
  0: license_plate
```

---

# 🤖 YOLOv8 Model

A custom YOLOv8 model was trained specifically for license plate detection.

The trained model detects the location of the license plate before sending the cropped region to EasyOCR.

The detection process is:

```text
Input Image
     ↓
YOLOv8
     ↓
License Plate Bounding Box
     ↓
Crop License Plate
     ↓
Send Crop to OCR
```

---

# 📊 Model Validation Results

The trained YOLOv8 model achieved the following validation results:

| Metric | Result |
|---|---:|
| Precision | 96.4% |
| Recall | 92.3% |
| mAP@50 | 97.0% |
| mAP@50-95 | 57.7% |

### Metric Explanation

**Precision – 96.4%**

Measures how many of the detected license plates were actually correct.

**Recall – 92.3%**

Measures how many of the actual license plates were successfully detected.

**mAP@50 – 97.0%**

Measures detection performance using an IoU threshold of 0.50.

**mAP@50-95 – 57.7%**

Measures stricter bounding-box localization performance across multiple IoU thresholds.

> These results are based on the project's validation dataset. Real-world performance may vary depending on lighting, camera quality, plate size, blur, occlusion, viewing angle, and weather conditions.

---

# 🖼️ RUN ANPR ON DATASET

The complete ANPR pipeline can process the dataset images automatically.

Run:

```powershell
python app.py --source "data/images 01/video_images" --output outputs
```

The application processes the images through:

```text
Dataset Image
      ↓
Fog Detection
      ↓
Image Enhancement
      ↓
YOLOv8
      ↓
License Plate Detection
      ↓
License Plate Crop
      ↓
OCR Preprocessing
      ↓
EasyOCR
      ↓
Plate Number
      ↓
Save Result
```

---

# 📁 Dataset Output

After processing, the results are stored in:

```text
outputs/
│
├── anpr_results.csv
│
└── annotated/
    ├── image_1.jpg
    ├── image_2.jpg
    ├── image_3.jpg
    └── ...
```

The annotated images contain:

- License plate bounding boxes
- Detected plate text
- Detection information
- OCR information
- Fog/visibility information where applicable

---

# 📊 CSV Results

The ANPR pipeline generates:

```text
outputs/anpr_results.csv
```

The CSV contains the processed recognition results.

Open the CSV using:

```powershell
start outputs\anpr_results.csv
```

Open the output directory:

```powershell
explorer outputs
```

---

# 📹 REAL-TIME ANPR

The project supports real-time license plate recognition using a webcam.

## Run

Connect a webcam and execute:

```powershell
python realtime_anpr.py
```

The system opens a real-time computer vision window.

---

# 🎥 Real-Time Processing Pipeline

```text
Webcam Frame
     ↓
Fog Detection
     ↓
Image Enhancement
     ↓
YOLOv8 Detection
     ↓
License Plate Crop
     ↓
OCR Preprocessing
     ↓
EasyOCR
     ↓
Plate Number
     ↓
Display Result
```

---

# 🖥️ Real-Time Display

The real-time window can display:

- License plate bounding box
- Detected license plate number
- YOLO detection confidence
- OCR confidence
- Fog status
- Processed video frame

The default webcam is:

```text
Camera Index = 0
```

---

# ⛔ Stop Real-Time ANPR

Press:

```text
Q
```

to close the application.

---

# 🎥 VIDEO ANPR

The system can process recorded vehicle videos.

Run:

```powershell
python video_anpr.py --source "input_video.mp4"
```

Replace:

```text
input_video.mp4
```

with the path to your video.

Example:

```powershell
python video_anpr.py --source "videos/traffic.mp4"
```

---

# 🌫️ FOG DETECTION

The project includes a fog/visibility detection module.

The fog detector analyzes image properties including:

- Image sharpness
- Contrast
- Saturation
- Local image characteristics

The system estimates whether the input image has reduced visibility.

Workflow:

```text
Input Image
     ↓
Visibility Analysis
     ↓
Sharpness Analysis
     ↓
Contrast Analysis
     ↓
Saturation Analysis
     ↓
Fog Score
     ↓
Clear / Foggy
```

---

# 🌫️ Fog Enhancement

If foggy conditions are detected, image enhancement can be applied before OCR.

```text
Input Image
     ↓
Fog Detection
     ↓
Foggy?
   /     \
 Yes      No
  ↓        ↓
Enhance   Original
  \        /
   \      /
    ↓    ↓
 YOLOv8 + OCR
```

This helps improve the quality of the image under reduced visibility.

---

# 🔤 OCR PROCESSING

EasyOCR is used to recognize the characters present on the detected license plate.

The OCR pipeline includes:

```text
License Plate Crop
        ↓
Resize
        ↓
Grayscale Conversion
        ↓
Contrast Enhancement
        ↓
Noise Reduction
        ↓
EasyOCR
        ↓
Character Filtering
        ↓
License Plate Number
```

---

# 🔍 License Plate Detection

YOLOv8 first identifies the license plate region.

For example:

```text
Vehicle Image
      ↓
YOLOv8
      ↓
┌─────────────────────┐
│     LICENSE PLATE   │
└─────────────────────┘
```

Only the detected license plate region is passed to the OCR engine.

This is more effective than applying OCR directly to the complete vehicle image.

---

# 🔡 License Plate Text Processing

After OCR, the detected text is processed to remove unnecessary characters.

The system can use an Indian-style license plate pattern such as:

```text
XX00XX0000
```

Example:

```text
MH01DE2780
```

The OCR output is filtered to retain relevant alphanumeric characters.

---

# 🧪 TRAIN THE YOLOv8 MODEL

If you want to retrain the license plate detector:

```powershell
python train_plate.py --data plate_data.yaml --epochs 80 --imgsz 640 --device cpu
```

### Training Parameters

| Parameter | Value |
|---|---|
| Model | YOLOv8 |
| Image Size | 640 × 640 |
| Epochs | Up to 80 |
| Device | CPU |
| Classes | 1 |
| Class | license_plate |

The training results are generated in the YOLO `runs` directory.

The best trained model is stored under the training output directory.

---

# 🧪 Model Validation

YOLOv8 validation can be used to evaluate the trained model.

The evaluation measures:

```text
Precision
Recall
mAP@50
mAP@50-95
```

These metrics help evaluate how accurately the model detects license plates.

---

# 🖼️ TEST A SINGLE IMAGE

A single image can be processed using:

```powershell
python app.py --source "data/images 01/video_images/car-wbs-MH01DE2780_00000.png" --output outputs/demo
```

The processed result will be saved under:

```text
outputs/demo/
```

---

# 👁️ VIEW AN IMAGE DIRECTLY

To display a test image using OpenCV:

```powershell
python -c "import cv2; img=cv2.imread('data/images 01/video_images/car-wbs-MH01DE2780_00000.png'); cv2.imshow('ANPR Image',img); cv2.waitKey(0); cv2.destroyAllWindows()"
```

---

# 📋 IMPORTANT COMMANDS

## Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Run Dataset ANPR

```powershell
python app.py --source "data/images 01/video_images" --output outputs
```

## Run Single Image

```powershell
python app.py --source "data/images 01/video_images/car-wbs-MH01DE2780_00000.png" --output outputs/demo
```

## Run Real-Time Webcam

```powershell
python realtime_anpr.py
```

## Run Video

```powershell
python video_anpr.py --source "input_video.mp4"
```

## Train YOLOv8

```powershell
python train_plate.py --data plate_data.yaml --epochs 80 --imgsz 640 --device cpu
```

## Open Output Folder

```powershell
explorer outputs
```

## Open CSV Results

```powershell
start outputs\anpr_results.csv
```

---

# ⚠️ Important Notes

## 1. Dataset Location

The dataset should be available at:

```text
data/images 01/video_images
```

If the dataset is stored in another location, change the source path:

```powershell
python app.py --source "YOUR_DATASET_PATH" --output outputs
```

---

## 2. Trained Model

The real-time system requires:

```text
models/plate_best.pt
```

Make sure the model exists before running:

```powershell
python realtime_anpr.py
```

---

## 3. Webcam

The default webcam index is:

```text
0
```

If the webcam does not open:

- Check the webcam connection
- Check Windows camera permissions
- Close other applications using the camera
- Try another camera index if supported

---

## 4. CPU Execution

The project supports CPU execution.

However, real-time processing may be slower on CPU compared with a CUDA-enabled GPU.

The project was developed and tested using CPU-based execution.

---

# 📊 Expected Output

The final system produces:

```text
Input
  ↓
License Plate Detected
  ↓
Plate Region Cropped
  ↓
Fog Condition Analyzed
  ↓
Image Enhanced if Required
  ↓
OCR Applied
  ↓
Plate Text Extracted
  ↓
Result Displayed
  ↓
Result Saved
```

Example:

```text
Detected Plate:
MH01DE2780

Detection Confidence:
0.XX

OCR Confidence:
0.XX

Fog Status:
Clear / Foggy
```

> OCR confidence can vary significantly depending on plate size, image quality, blur, lighting, viewing angle, and visibility conditions.

---

# 🚗 Applications

The ANPR system can be applied to:

- Smart parking systems
- Vehicle access control
- Traffic monitoring
- Campus security
- Toll management
- Highway surveillance
- Smart city systems
- Automated vehicle identification
- Parking management
- Security surveillance
- Restricted-area vehicle monitoring
- Gate-entry management

---

# 🔮 Future Enhancements

Possible future improvements include:

- Multi-vehicle tracking
- Advanced image dehazing
- License plate super-resolution
- Night-time enhancement
- Vehicle make and model recognition
- Vehicle speed estimation
- Database integration
- Web-based ANPR dashboard
- Cloud deployment
- Edge-device deployment
- Unauthorized vehicle alerts
- Historical vehicle tracking
- Automatic violation detection
- Real-time alert notifications
- Centralized vehicle database

---

# 🔐 Privacy and Responsible Use

License plate recognition can involve sensitive vehicle-related information.

When deploying the system in real-world environments:

- Follow applicable privacy and data-protection regulations.
- Use collected data only for authorized purposes.
- Restrict access to stored ANPR results.
- Protect stored images and license plate information.
- Avoid unnecessary retention of vehicle data.
- Obtain required permissions for surveillance deployments.

This project is primarily intended for academic, educational, and research purposes.

---

# 📈 Project Status

✅ Dataset Preparation
✅ License Plate Annotation
✅ YOLOv8 Training
✅ YOLOv8 Validation
✅ License Plate Detection
✅ EasyOCR Integration
✅ OCR Preprocessing
✅ Fog Detection
✅ Image Enhancement
✅ Dataset ANPR
✅ Real-Time ANPR
✅ Video ANPR
✅ CSV Result Generation
✅ GitHub Documentation


---

# 📌 Project Highlights

### Deep Learning

Custom YOLOv8 model trained specifically for license plate detection.

### Computer Vision

OpenCV is used for:

- Image processing
- Video processing
- Frame handling
- Image enhancement
- Real-time visualization

### OCR

EasyOCR is integrated to extract characters from detected license plates.

### Visibility Analysis

A fog detection module analyzes image quality and determines whether enhancement is required.

### Real-Time Processing

The system can process live webcam frames and display ANPR results in real time.

### Automated Results

Detected information can be stored in CSV format along with annotated images.

---

# 📜 License

This project is developed for **academic, educational, and research purposes**.

---

# 👩‍💻 Author

## Dhanushini S

### Automatic Number Plate Recognition Using YOLOv8, EasyOCR and Fog Detection

**Domain:**

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Optical Character Recognition
- Intelligent Transportation Systems

---

# ⭐ Acknowledgements

This project uses open-source technologies and libraries including:

- Ultralytics YOLOv8
- EasyOCR
- OpenCV
- PyTorch
- NumPy
- Pandas

---

# ⭐ Project Summary

This project combines **YOLOv8-based license plate detection, EasyOCR-based character recognition, OpenCV image processing, and fog/visibility analysis** into a single ANPR pipeline.

The system supports **dataset processing, real-time webcam recognition, and video-based vehicle monitoring**, providing an end-to-end computer vision solution for automated license plate recognition.

```text
YOLOv8
   +
EasyOCR
   +
OpenCV
   +
Fog Detection
   ↓
Intelligent ANPR System
```
