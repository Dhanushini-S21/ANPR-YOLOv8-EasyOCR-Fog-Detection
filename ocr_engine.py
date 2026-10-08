import re
import cv2
import numpy as np
import easyocr


class PlateOCR:
    def __init__(self, languages=None, gpu=False):
        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    @staticmethod
    def clean_text(text):
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        # OCR commonly confuses these characters.
        text = text.replace(" ", "")
        return text

    @staticmethod
    def valid_indian_plate(text):
        # Flexible validation for common Indian private/commercial formats.
        patterns = [
            r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
            r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$",
        ]
        return any(re.match(p, text) for p in patterns)

    @staticmethod
    def preprocess(plate):
        if plate is None or plate.size == 0:
            return None

        h, w = plate.shape[:2]
        scale = max(2.5, 320 / max(w, 1))
        resized = cv2.resize(
            plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 7, 40, 40)

        return gray

    def read(self, plate):
        processed = self.preprocess(plate)
        if processed is None:
            return {"text": "", "confidence": 0.0, "valid": False}

        variants = [
            processed,
            cv2.threshold(
                processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],
        ]

        best = {"text": "", "confidence": 0.0, "valid": False}

        for img in variants:
            results = self.reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                text_threshold=0.35,
                low_text=0.20,
                link_threshold=0.20,
            )

            for _, text, conf in results:
                cleaned = self.clean_text(text)
                if len(cleaned) < 4:
                    continue

                score = float(conf)
                if self.valid_indian_plate(cleaned):
                    score += 0.25

                if score > best["confidence"]:
                    best = {
                        "text": cleaned,
                        "confidence": round(min(score, 1.0), 4),
                        "valid": self.valid_indian_plate(cleaned),
                    }

        return best
