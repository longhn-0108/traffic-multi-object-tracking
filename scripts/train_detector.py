"""Train a YOLO detector on a YOLO-format dataset."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on UA-DETRAC.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base YOLO weights.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="e.g. 0, 0,1, or cpu.")
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="vehicle_detector")
    args = parser.parse_args()

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print(results)


if __name__ == "__main__":
    main()
