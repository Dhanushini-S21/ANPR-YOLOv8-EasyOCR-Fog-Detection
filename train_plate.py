import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="plate_data.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="plate_yolov8",
        pretrained=True,
        optimizer="auto",
        patience=20,
        cache=False,
        plots=True,
        degrees=5.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.0,
        mosaic=0.5,
    )

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device
    )
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
