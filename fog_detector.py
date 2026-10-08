import cv2
import numpy as np

class FogDetector:
    """
    Analytical visibility/fog detector.
    It does not pretend to be a trained fog classifier.
    The score combines:
      - Laplacian sharpness
      - grayscale contrast
      - saturation
      - local contrast
    """

    def __init__(self, fog_threshold=0.42):
        self.fog_threshold = fog_threshold

    @staticmethod
    def _normalize(value, low, high):
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    def analyze(self, image):
        if image is None or image.size == 0:
            return {"status": "unknown", "score": 0.0}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = float(gray.std())
        saturation = float(hsv[:, :, 1].mean())

        # CLAHE gives a useful local-contrast proxy.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        local = clahe.apply(gray)
        local_gain = float(np.std(local) - contrast)

        # Lower sharpness/contrast/saturation generally indicates visibility loss.
        blur_component = 1.0 - self._normalize(sharpness, 20.0, 500.0)
        contrast_component = 1.0 - self._normalize(contrast, 15.0, 75.0)
        saturation_component = 1.0 - self._normalize(saturation, 15.0, 100.0)
        local_component = self._normalize(local_gain, 0.0, 30.0)

        score = (
            0.40 * blur_component +
            0.30 * contrast_component +
            0.20 * saturation_component +
            0.10 * local_component
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score >= 0.65:
            status = "heavy_fog"
        elif score >= self.fog_threshold:
            status = "foggy"
        elif score >= 0.30:
            status = "hazy"
        else:
            status = "clear"

        return {
            "status": status,
            "score": round(score, 4),
            "sharpness": round(float(sharpness), 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
        }

    def enhance(self, image, fog_status):
        if fog_status not in {"foggy", "heavy_fog", "hazy"}:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # Mild denoising + unsharp mask.
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        blur = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        sharp = cv2.addWeighted(denoised, 1.35, blur, -0.35, 0)
        return sharp
