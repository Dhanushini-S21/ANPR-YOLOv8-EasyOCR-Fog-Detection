# Automatic Number Plate Recognition (ANPR)
## YOLOv8 + EasyOCR + Fog Detection

This project is designed for the supplied vehicle/number-plate image datasets.

### What the pipeline does

1. YOLOv8 detects vehicles.
2. A plate-localization module searches the vehicle ROI for the license plate.
3. The fog module estimates visibility degradation using contrast, sharpness and saturation.
4. If visibility is poor, CLAHE/denoising/sharpening is applied before OCR.
5. EasyOCR reads the Indian registration number.
6. OCR output is cleaned and validated against Indian-style plate patterns.
7. Results are saved as annotated images and CSV.
8. Video mode processes complete videos frame-by-frame.
9. Optional custom YOLOv8 plate training is included.

### Important dataset note

The supplied ZIP files contain images only; no YOLO `.txt` label files were found.
Therefore, `plate_best.pt` cannot be legitimately trained from these images until
license-plate bounding boxes are annotated.

The default pipeline is still runnable because it uses:
- pretrained YOLOv8 for vehicle detection
- OpenCV plate localization fallback
- EasyOCR for text recognition
- analytical fog/visibility detection

For a production-grade plate detector, annotate plates and run `train_plate.py`.

## 1. Folder structure

After extracting the project:

ANPR_YOLOv8_EasyOCR_Fog/
  app.py
  video_anpr.py
  plate_detector.py
  fog_detector.py
  ocr_engine.py
  utils.py
  annotate.py
  train_plate.py
  requirements.txt
  data/
    images/
    labels/
  models/
    plate_best.pt        # optional custom YOLOv8 plate model
  outputs/
  README.md

Put your supplied images under `data/images/`.

## 2. Installation

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

EasyOCR may download its recognition model the first time it runs.

## 3. Run on the supplied images

python app.py --source data/images --output outputs

The first YOLOv8 run automatically downloads `yolov8n.pt`.

For GPU:

python app.py --source data/images --output outputs --device 0

For CPU:

python app.py --source data/images --output outputs --device cpu

## 4. Run on a video

python video_anpr.py --source your_video.mp4 --output outputs/anpr_video.mp4

GPU:

python video_anpr.py --source your_video.mp4 --output outputs/anpr_video.mp4 --device 0

## 5. Custom plate detector (recommended)

The correct production workflow is:

A. Annotate the license plate boxes:

python annotate.py --images data/images --labels data/labels

The annotation window lets you draw one or more plate boxes.
Press ENTER to save an image, ESC to skip, Q to quit.

B. Split data into:

data/plate_dataset/
  images/train
  images/val
  labels/train
  labels/val

C. Create/update `plate_data.yaml` and train:

python train_plate.py --data plate_data.yaml --epochs 80 --imgsz 640

The best model will be saved under:
runs/detect/plate_yolov8/weights/best.pt

Copy that file to:

models/plate_best.pt

Then run:

python app.py --source data/images --output outputs --plate-model models/plate_best.pt

This replaces the OpenCV plate-localization fallback with a learned YOLOv8
license-plate detector.

## 6. Output

`outputs/anpr_results.csv` contains:
- image
- vehicle_class
- vehicle_confidence
- plate_text
- ocr_confidence
- fog_status
- fog_score
- plate_bbox
- processing_status

Annotated images are written to `outputs/annotated/`.

## 7. Architecture

Input Image/Video
       |
       v
YOLOv8 Vehicle Detection
       |
       v
Vehicle ROI
       |
       +----> Fog / Visibility Estimation
       |             |
       |             v
       |       Image Enhancement
       |
       v
Plate Detection
  (custom YOLOv8 if available,
   otherwise OpenCV fallback)
       |
       v
Plate Crop
       |
       v
EasyOCR
       |
       v
Text Cleaning + Indian Plate Validation
       |
       v
ANPR Result + CSV + Annotated Frame

## 8. Accuracy note

For serious deployment, train a custom plate detector using annotated plate
bounding boxes. OCR quality also depends strongly on plate resolution, angle,
motion blur and lighting. Fog detection in this project is a visibility
estimator, not a falsely claimed pretrained fog classifier.
