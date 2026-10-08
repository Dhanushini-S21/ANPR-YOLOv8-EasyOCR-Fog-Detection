import cv2
import re
import time
import easyocr
from ultralytics import YOLO

from fog_detector import FogDetector


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/plate_best.pt"

# YOLO confidence threshold
YOLO_CONFIDENCE = 0.35

# OCR confidence threshold
OCR_CONFIDENCE = 0.20

# OCR every N frames to improve real-time performance
OCR_INTERVAL = 5

# Camera index
CAMERA_INDEX = 0


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLOv8 license plate model...")
model = YOLO(MODEL_PATH)

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

print("Loading fog detector...")
fog_detector = FogDetector()

print("All models loaded successfully.")


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_plate(plate):
    """
    Prepare the detected license plate for OCR.
    """

    if plate is None or plate.size == 0:
        return None

    # Upscale plate
    plate = cv2.resize(
        plate,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Mild denoising
    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return enhanced


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_plate_text(text):
    """
    Clean OCR output and keep alphanumeric characters.
    """

    text = text.upper()

    # Remove spaces and special characters
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# INDIAN NUMBER PLATE VALIDATION
# ============================================================

def is_valid_indian_plate(text):
    """
    Basic validation for Indian registration numbers.

    Examples:
    MH01DE2780
    TN38AB1234
    KA01MN5678
    """

    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"

    return bool(re.match(pattern, text))


# ============================================================
# OCR FUNCTION
# ============================================================

def recognize_plate(plate):

    processed = preprocess_plate(plate)

    if processed is None:
        return "", 0.0

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    candidates = []

    for result in results:

        text = result[1]
        confidence = float(result[2])

        text = clean_plate_text(text)

        if not text:
            continue

        candidates.append(
            (text, confidence)
        )

    if not candidates:
        return "", 0.0

    # Prefer valid Indian registration format
    valid_candidates = [
        item for item in candidates
        if is_valid_indian_plate(item[0])
    ]

    if valid_candidates:
        valid_candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return valid_candidates[0]

    # Otherwise return highest confidence
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[0]


# ============================================================
# START CAMERA
# ============================================================

print("\nStarting webcam...")
print("Press Q to quit.")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    print("\nTry changing:")
    print("CAMERA_INDEX = 0")
    print("to")
    print("CAMERA_INDEX = 1")

    raise SystemExit


# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# REAL-TIME VARIABLES
# ============================================================

frame_count = 0

last_plate_text = ""
last_ocr_confidence = 0.0

last_fog_status = "unknown"
last_fog_score = 0.0

last_ocr_time = 0


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame.")

        break

    frame_count += 1

    display_frame = frame.copy()


    # --------------------------------------------------------
    # FOG ANALYSIS
    # --------------------------------------------------------

    try:

        fog_result = fog_detector.analyze(frame)

        last_fog_status = fog_result.get(
            "status",
            "unknown"
        )

        last_fog_score = float(
            fog_result.get(
                "score",
                0.0
            )
        )

    except Exception as e:

        last_fog_status = "unknown"
        last_fog_score = 0.0


    # --------------------------------------------------------
    # YOLO LICENSE PLATE DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    detected_plate = False


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))

            x2 = min(
                frame.shape[1],
                int(x2)
            )

            y2 = min(
                frame.shape[0],
                int(y2)
            )


            if x2 <= x1 or y2 <= y1:
                continue


            detected_plate = True


            # ------------------------------------------------
            # CROP LICENSE PLATE
            # ------------------------------------------------

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            current_time = time.time()

            if (
                frame_count % OCR_INTERVAL == 0
                and current_time - last_ocr_time > 0.1
            ):

                text, confidence = recognize_plate(
                    plate_crop
                )

                if text:

                    last_plate_text = text
                    last_ocr_confidence = confidence

                last_ocr_time = current_time


            # ------------------------------------------------
            # DRAW PLATE BOX
            # ------------------------------------------------

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DISPLAY PLATE TEXT
            # ------------------------------------------------

            if last_plate_text:

                label = (
                    f"{last_plate_text} "
                    f"({last_ocr_confidence:.2f})"
                )

            else:

                label = "Plate detected"


            # Background for label
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )[0]


            cv2.rectangle(
                display_frame,
                (x1, max(0, y1 - 35)),
                (
                    x1 + text_size[0] + 10,
                    y1
                ),
                (0, 255, 0),
                -1
            )


            cv2.putText(
                display_frame,
                label,
                (x1 + 5, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        display_frame,
        (10, 10),
        (430, 145),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        display_frame,
        "REAL-TIME ANPR",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    # Plate
    plate_display = (
        last_plate_text
        if last_plate_text
        else "Not detected"
    )


    cv2.putText(
        display_frame,
        f"Plate: {plate_display}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # OCR confidence
    cv2.putText(
        display_frame,
        f"OCR: {last_ocr_confidence:.2f}",
        (25, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # Fog
    cv2.putText(
        display_frame,
        f"Visibility: {last_fog_status}",
        (25, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Real-Time ANPR - YOLOv8 + EasyOCR + Fog Detection",
        display_frame
    )


    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# RELEASE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nReal-time ANPR stopped.")