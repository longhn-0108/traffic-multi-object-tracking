"""Run YOLO detection on an image/video source."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(source=args.source, conf=args.conf, save=args.save)


if __name__ == "__main__":
    main()
