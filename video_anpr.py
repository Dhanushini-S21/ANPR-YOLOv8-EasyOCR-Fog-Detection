import argparse
import os
import cv2
import torch
from ultralytics import YOLO

from fog_detector import FogDetector
from plate_detector import PlateDetector
from ocr_engine import PlateOCR
from utils import crop_box, ensure_dir


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/anpr_video.mp4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vehicle-model", default="yolov8n.pt")
    parser.add_argument("--plate-model", default="models/plate_best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-gpu", action="store_true")
    parser.add_argument("--ocr-every", type=int, default=5)
    args = parser.parse_args()

    device = args.device
    if device == "0" and not torch.cuda.is_available():
        device = "cpu"

    ensure_dir(os.path.dirname(args.output) or ".")

    model = YOLO(args.vehicle_model)
    model.to(device if device != "cpu" else "cpu")

    plate_model = args.plate_model if os.path.exists(args.plate_model) else None
    plate_detector = PlateDetector(plate_model, conf=args.conf)
    fog_detector = FogDetector()
    ocr = PlateOCR(gpu=args.ocr_gpu)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_no = 0
    last_text = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        results = model.predict(source=frame, conf=args.conf, verbose=False)

        if results:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                roi = crop_box(frame, (x1, y1, x2, y2))
                if roi is None:
                    continue

                fog = fog_detector.analyze(roi)
                enhanced = fog_detector.enhance(roi, fog["status"])

                text = last_text.get((x1//20, y1//20), "READING...")
                if frame_no % max(args.ocr_every, 1) == 0:
                    candidates = plate_detector.detect(enhanced)
                    best = {"text": "", "confidence": 0.0}
                    for p in candidates:
                        crop = crop_box(enhanced, p)
                        r = ocr.read(crop)
                        if r["confidence"] > best["confidence"]:
                            best = r
                    if best["text"]:
                        text = best["text"]
                    last_text[(x1//20, y1//20)] = text

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                label = f"{VEHICLE_CLASSES[cls_id]} | {text} | {fog['status']}"
                cv2.putText(
                    frame, label, (x1, max(25, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA
                )

        writer.write(frame)

    cap.release()
    writer.release()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
