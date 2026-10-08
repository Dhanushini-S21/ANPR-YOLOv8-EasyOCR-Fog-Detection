import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PlateDetector:
    """
    Preferred mode:
      custom YOLOv8 license-plate detector.

    Fallback mode:
      OpenCV contour/edge-based plate localization inside vehicle ROIs.
    """

    def __init__(self, model_path=None, conf=0.25):
        self.conf = conf
        self.model = None
        if model_path and os.path.exists(model_path):
            if YOLO is None:
                raise ImportError("Install ultralytics first.")
            self.model = YOLO(model_path)

    @staticmethod
    def _fallback(vehicle_roi):
        h, w = vehicle_roi.shape[:2]
        if h < 20 or w < 20:
            return []

        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 50, 50)
        edges = cv2.Canny(gray, 80, 180)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < 0.002 * w * h:
                continue

            ratio = cw / max(ch, 1)
            fill = area / max(w * h, 1)

            # Indian plates are normally wider than tall.
            if 2.0 <= ratio <= 7.5 and 0.005 <= fill <= 0.35:
                score = (
                    min(ratio / 4.5, 1.0) * 0.35 +
                    min(area / (0.08 * w * h), 1.0) * 0.35 +
                    min(cw / max(w, 1), 1.0) * 0.30
                )
                candidates.append((score, (x, y, x + cw, y + ch)))

        candidates.sort(reverse=True, key=lambda z: z[0])
        return [box for _, box in candidates[:5]]

    def detect(self, vehicle_roi):
        if self.model is None:
            return self._fallback(vehicle_roi)

        results = self.model.predict(
            source=vehicle_roi, conf=self.conf, verbose=False
        )
        boxes = []
        if not results:
            return boxes

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            boxes.append(tuple(xyxy))
        return boxes
