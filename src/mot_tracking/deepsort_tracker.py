"""YOLO + DeepSORT vehicle tracking on video inputs.

The tracker writes optional annotated videos and MOTChallenge/TrackEval-style
tracking files. Dataset videos and model weights are intentionally external.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track vehicles with YOLO and DeepSORT."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", required=True, help="YOLO weights path.")
    parser.add_argument(
        "--output",
        default="outputs/videos/deepsort.mp4",
        help="Annotated video output path.",
    )
    parser.add_argument(
        "--trackeval-output",
        default="outputs/trackeval/deepsort.txt",
        help="MOTChallenge/TrackEval result path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum YOLO detection confidence.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save an annotated video.",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Display tracking in a local OpenCV window.",
    )
    parser.add_argument(
        "--no-save-txt",
        action="store_true",
        help="Disable TrackEval/MOTChallenge output.",
    )
    return parser.parse_args()


def run_tracking(
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path = "outputs/videos/deepsort.mp4",
    trackeval_output: str | Path = "outputs/trackeval/deepsort.txt",
    conf: float = 0.25,
    save_video: bool = False,
    show_video: bool = False,
    save_trackeval: bool = True,
) -> None:
    model = YOLO(str(model_path))
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
    )

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if save_video:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

    track_file = None
    if save_trackeval:
        trackeval_output = Path(trackeval_output)
        trackeval_output.parent.mkdir(parents=True, exist_ok=True)
        track_file = trackeval_output.open("w", encoding="utf-8")

    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            results = model(frame, conf=conf, verbose=False)

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    detections.append(
                        ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         confidence,
                         class_name)
                    )

            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed() or track.track_id is None:
                    continue

                try:
                    track_id = int(track.track_id)
                except (TypeError, ValueError):
                    continue

                confidence = track.get_det_conf()
                if confidence is None:
                    continue

                ltrb = track.to_ltrb(orig=True)
                if ltrb is None:
                    ltrb = track.to_ltrb()

                x1, y1, x2, y2 = map(int, ltrb)
                width_box = x2 - x1
                height_box = y2 - y1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                )

                if track_file is not None:
                    # MOTChallenge format:
                    # frame,id,x,y,w,h,confidence,class,visibility,unused
                    track_file.write(
                        f"{frame_number},{track_id},{x1},{y1},"
                        f"{width_box},{height_box},{float(confidence):.6f},"
                        "1,1,-1\n"
                    )

            if writer is not None:
                writer.write(frame)

            if show_video:
                cv2.imshow("YOLO + DeepSORT", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if track_file is not None:
            track_file.close()
        if show_video:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_tracking(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        trackeval_output=args.trackeval_output,
        conf=args.conf,
        save_video=args.save_video,
        show_video=args.show_video,
        save_trackeval=not args.no_save_txt,
    )


if __name__ == "__main__":
    main()
