import argparse
import os
import cv2
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Draw license-plate boxes and save YOLO labels."
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    ensure_dir(args.labels)

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = sorted([
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if f.lower().endswith(extensions)
    ])

    print("Controls: draw ROI -> ENTER saves, ESC skips, Q quits.")
    for path in images:
        image = cv2.imread(path)
        if image is None:
            continue

        h, w = image.shape[:2]
        clone = image.copy()

        cv2.putText(
            clone, "Draw PLATE box. ENTER=save ESC=skip Q=quit",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2
        )

        box = cv2.selectROI("Plate Annotation", clone, fromCenter=False)
        cv2.destroyWindow("Plate Annotation")

        x, y, bw, bh = map(int, box)
        if bw <= 0 or bh <= 0:
            continue

        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        label_path = os.path.join(
            args.labels, os.path.splitext(os.path.basename(path))[0] + ".txt"
        )

        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        print("Saved:", label_path)

        key = cv2.waitKey(1)
        if key in (ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
