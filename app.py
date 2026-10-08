import argparse
import os
import cv2
import torch
from ultralytics import YOLO

from fog_detector import FogDetector
from plate_detector import PlateDetector
from ocr_engine import PlateOCR
from utils import ensure_dir, crop_box, save_csv


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def process_image(
    image_path,
    vehicle_model,
    plate_detector,
    fog_detector,
    ocr,
    output_path,
    conf=0.25,
):
    image = cv2.imread(image_path)
    if image is None:
        return []

    annotated = image.copy()
    results = vehicle_model.predict(
        source=image, conf=conf, verbose=False
    )

    rows = []

    if not results:
        cv2.imwrite(output_path, annotated)
        return rows

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        if cls_id not in VEHICLE_CLASSES:
            continue

        vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
        vehicle_roi = crop_box(image, (vx1, vy1, vx2, vy2))
        if vehicle_roi is None:
            continue

        fog = fog_detector.analyze(vehicle_roi)
        enhanced_vehicle = fog_detector.enhance(vehicle_roi, fog["status"])

        plate_boxes = plate_detector.detect(enhanced_vehicle)

        best_ocr = {"text": "", "confidence": 0.0, "valid": False}
        best_plate = None

        for pbox in plate_boxes:
            plate_crop = crop_box(enhanced_vehicle, pbox)
            result = ocr.read(plate_crop)

            if result["confidence"] > best_ocr["confidence"]:
                best_ocr = result
                best_plate = pbox

        if best_plate:
            px1, py1, px2, py2 = best_plate
            abs_box = (
                vx1 + px1, vy1 + py1,
                vx1 + px2, vy1 + py2
            )
            cv2.rectangle(
                annotated,
                (abs_box[0], abs_box[1]),
                (abs_box[2], abs_box[3]),
                (0, 255, 0), 2
            )
        else:
            abs_box = None

        cv2.rectangle(
            annotated, (vx1, vy1), (vx2, vy2), (255, 180, 0), 2
        )

        label = (
            f"{VEHICLE_CLASSES[cls_id]} {score:.2f} | "
            f"{best_ocr['text'] or 'PLATE NOT READ'} | "
            f"{fog['status']}"
        )
        cv2.putText(
            annotated, label, (vx1, max(25, vy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            cv2.LINE_AA
        )

        rows.append({
            "image": os.path.basename(image_path),
            "vehicle_class": VEHICLE_CLASSES[cls_id],
            "vehicle_confidence": round(score, 4),
            "plate_text": best_ocr["text"],
            "ocr_confidence": best_ocr["confidence"],
            "plate_valid_indian_format": best_ocr["valid"],
            "fog_status": fog["status"],
            "fog_score": fog["score"],
            "plate_bbox": abs_box,
        })

    cv2.imwrite(output_path, annotated)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vehicle-model", default="yolov8n.pt")
    parser.add_argument("--plate-model", default="models/plate_best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-gpu", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output)
    annotated_dir = os.path.join(args.output, "annotated")
    ensure_dir(annotated_dir)

    device = args.device
    if device == "0" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    print("Loading YOLOv8 vehicle model...")
    vehicle_model = YOLO(args.vehicle_model)
    vehicle_model.to(device if device != "cpu" else "cpu")

    print("Loading plate detector...")
    plate_detector = PlateDetector(
        args.plate_model if os.path.exists(args.plate_model) else None
    )
    if plate_detector.model is None:
        print("No custom plate model found. Using OpenCV fallback.")

    print("Loading EasyOCR...")
    ocr = PlateOCR(gpu=args.ocr_gpu)
    fog_detector = FogDetector()

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if os.path.isdir(args.source):
        image_paths = [
            os.path.join(args.source, f)
            for f in os.listdir(args.source)
            if f.lower().endswith(extensions)
        ]
    else:
        image_paths = [args.source]

    all_rows = []
    for i, image_path in enumerate(sorted(image_paths), 1):
        print(f"[{i}/{len(image_paths)}] {image_path}")
        out_name = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        out_path = os.path.join(annotated_dir, out_name)
        try:
            all_rows.extend(
                process_image(
                    image_path, vehicle_model, plate_detector,
                    fog_detector, ocr, out_path, args.conf
                )
            )
        except Exception as exc:
            print("ERROR:", exc)

    csv_path = os.path.join(args.output, "anpr_results.csv")
    save_csv(all_rows, csv_path)
    print(f"\nCompleted. Results: {csv_path}")
    print(f"Annotated images: {annotated_dir}")


if __name__ == "__main__":
    main()
