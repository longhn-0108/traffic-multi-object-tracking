from pathlib import Path
import argparse
import csv
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2

_YOLO_CONFIG_DIR = Path(__file__).resolve().parents[2] / ".ultralytics_config"
_YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))

from ultralytics import YOLO
try:
    from .association import StableIdStitcher
except ImportError:  # pragma: no cover - direct script execution
    from association import StableIdStitcher


VEHICLE_CLASSES = [0]  # COCO: car=2, bus=5
REAL_BOX_COLOR = (255, 0, 0)
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.8
DEFAULT_STITCH_MAX_GAP = 15
DEFAULT_STITCH_IOU = 0.25
DEFAULT_STITCH_CENTER_DIST = 0.6
DEFAULT_INPUT_DIR = "data/videos"
DEFAULT_VIDEO_OUTPUT_DIR = "outputs/videos/bytetrack"
DEFAULT_TRACKEVAL_TRACKER_DIR = "outputs/trackeval/bytetrack"
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
TRACKEVAL_CLASS_ID = 1
TRACKEVAL_VISIBILITY = -1


def normalize_sequence_name(sequence_name):
    sequence_name = Path(str(sequence_name)).stem
    if sequence_name.startswith("MVI_"):
        return sequence_name[4:]
    return sequence_name


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width - 1))
    y2 = max(y1 + 1, min(int(round(y2)), height - 1))
    return [x1, y1, x2, y2]


def draw_labeled_box(frame, box, label, color):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_h = text_h + baseline + 8
    label_w = text_w + 10

    label_y1 = max(0, y1 - label_h)
    label_y2 = label_y1 + label_h
    label_x2 = min(w - 1, x1 + label_w)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(frame, label, (x1 + 5, label_y2 - baseline - 4), font, font_scale, (255, 255, 255), thickness)


def write_trackeval_txt(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, removed_duplicates = deduplicate_trackeval_rows(rows)
    rows = sorted(rows, key=lambda row: (row["frame_num"], row["track_id"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            values = [
                row["frame_num"],
                row["track_id"],
                round(row["x"], 3),
                round(row["y"], 3),
                round(row["w"], 3),
                round(row["h"], 3),
                round(row["confidence"], 6),
                TRACKEVAL_CLASS_ID,
                TRACKEVAL_VISIBILITY,
            ]
            handle.write(",".join(str(value) for value in values) + "\n")
    return len(rows), removed_duplicates


class ByteTrack:
    def __init__(self, device=None, video_path=None, tracker_yaml=None, model_path=None):
        project_root = Path(__file__).resolve().parents[2]
        weight_path = Path(model_path) if model_path else project_root / "models" / "yolov11n_last.pt"
        self.video_path = Path(video_path) if video_path else project_root / "data" / "videos" / "39031.mp4"
        self.bytetrack_yaml = Path(tracker_yaml) if tracker_yaml else project_root / "configs" / "bytetrack.yaml"
        self.device = device

        if not weight_path.exists():
            raise FileNotFoundError(f"Model weight not found: {weight_path}")
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.bytetrack_yaml.exists():
            raise FileNotFoundError(f"ByteTrack config not found: {self.bytetrack_yaml}")

        self.model = YOLO(weight_path)

    def detect(self):
        predict_args = dict(
            source=self.video_path, 
            show=True, 
            line_width=2, 
            conf=DEFAULT_CONF, 
            iou=DEFAULT_IOU,
            classes=VEHICLE_CLASSES,
            save=False,
        )
        if self.device:
            predict_args["device"] = self.device
        return self.model.predict(**predict_args)

    def track(
        self,
        output_path=None,
        show=False,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        image_scale=1,
        save_pred_csv=None,
        save_eval_csv=None,
        save_trackeval_txt=None,
        max_frames=None,
        write_video=True,
        save_csv=True,
        stitch_ids=True,
        stitch_max_gap=DEFAULT_STITCH_MAX_GAP,
        stitch_iou=DEFAULT_STITCH_IOU,
        stitch_center_dist=DEFAULT_STITCH_CENTER_DIST,
    ):
        output_path = Path(output_path) if output_path else Path(__file__).resolve().parent 
        if save_csv:
            if save_pred_csv is None:
                save_pred_csv = output_path.with_name(f"{output_path.stem}_prediction.csv")
            else:
                save_pred_csv = Path(save_pred_csv)
            if save_eval_csv is None:
                save_eval_csv = output_path.with_name(f"{output_path.stem}_evaluation.csv")
            else:
                save_eval_csv = Path(save_eval_csv)
        else:
            save_pred_csv = None
            save_eval_csv = None

        frame_count = 0
        pred_rows = []
        eval_rows = []
        trackeval_rows = []
        sequence_name = normalize_sequence_name(self.video_path.stem)
        cap = cv2.VideoCapture(str(self.video_path))
        writer = None
        stitcher = StableIdStitcher(
            enabled=stitch_ids,
            max_gap=stitch_max_gap,
            iou_threshold=stitch_iou,
            center_dist_threshold=stitch_center_dist,
        )

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames is not None and frame_count > max_frames:
                break

            h, w = frame.shape[:2]
            new_h = round(h / image_scale)
            new_w = round(w / image_scale)
            frame = cv2.resize(frame, (new_w, new_h))
            scale_x = w / new_w
            scale_y = h / new_h

            if write_video and writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (new_w, new_h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")

            track_args = dict(
                source=frame, 
                persist=True, 
                tracker=str(self.bytetrack_yaml),
                conf=conf,
                iou=iou,
                classes=VEHICLE_CLASSES,
                verbose=False,
            )
            if self.device:
                track_args["device"] = self.device

            results = self.model.track(**track_args)
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                current_raw_ids = {int(track_id) for track_id in ids}
                blocked_stable_ids = {
                    stitcher.raw_to_stable[raw_id]
                    for raw_id in current_raw_ids
                    if raw_id in stitcher.raw_to_stable
                }
                for box, raw_track_id, confidence, class_id in zip(xyxy, ids, confidences, class_ids):
                    raw_track_id = int(raw_track_id)
                    box = [float(value) for value in box]
                    stable_track_id = stitcher.resolve(
                        raw_track_id,
                        box,
                        class_id,
                        frame_count,
                        blocked_stable_ids=blocked_stable_ids,
                    )
                    blocked_stable_ids.add(stable_track_id)
                    if write_video or show:
                        label = f"ID {stable_track_id}"
                        draw_labeled_box(frame, box, label, REAL_BOX_COLOR)
                    if save_pred_csv or save_eval_csv or save_trackeval_txt:
                        original_box = [
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y,
                        ]
                        x1, y1, x2, y2 = original_box
                        base_row = {
                            "sequence": sequence_name,
                            "frame_num": frame_count,
                            "track_id": stable_track_id,
                            "confidence": round(float(confidence), 6),
                            "class_id": int(class_id),
                        }
                        if save_pred_csv:
                            pred_rows.append(
                                {
                                    **base_row,
                                    "x1": round(x1, 3),
                                    "y1": round(y1, 3),
                                    "x2": round(x2, 3),
                                    "y2": round(y2, 3),
                                }
                            )
                        if save_eval_csv:
                            eval_rows.append(
                                {
                                    **base_row,
                                    "x": round(x1, 3),
                                    "y": round(y1, 3),
                                    "w": round(x2 - x1, 3),
                                    "h": round(y2 - y1, 3),
                                }
                            )
                        if save_trackeval_txt:
                            trackeval_rows.append(
                                {
                                    "frame_num": frame_count,
                                    "track_id": stable_track_id,
                                    "x": float(x1),
                                    "y": float(y1),
                                    "w": float(x2 - x1),
                                    "h": float(y2 - y1),
                                    "confidence": float(confidence),
                                }
                            )

            if writer is not None:
                writer.write(frame)

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")

            if show:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        if save_pred_csv:
            save_pred_csv.parent.mkdir(parents=True, exist_ok=True)
            with save_pred_csv.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = ["sequence", "frame_num", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pred_rows)
            print(f"Saved prediction CSV to {save_pred_csv}")

        if save_eval_csv:
            save_eval_csv.parent.mkdir(parents=True, exist_ok=True)
            with save_eval_csv.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = ["sequence", "frame_num", "track_id", "x", "y", "w", "h", "confidence", "class_id"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(eval_rows)
            print(f"Saved evaluation CSV to {save_eval_csv}")

        if save_trackeval_txt:
            save_trackeval_txt = Path(save_trackeval_txt)
            row_count, removed_duplicates = write_trackeval_txt(save_trackeval_txt, trackeval_rows)
            print(f"Saved TrackEval TXT to {save_trackeval_txt}")
            print(f"TrackEval rows: {row_count}")
            if removed_duplicates:
                print(f"Removed duplicate TrackEval rows: {removed_duplicates}")

        print(f"Raw IDs: {stitcher.raw_count}")
        print(f"Stable IDs: {stitcher.stable_count}")
        print(f"Stitched ID fragments: {stitcher.stitched_fragments}")
        if write_video:
            print(f"Saved tracking video to {output_path}")
        else:
            print("Video output disabled")
        return output_path


def run_detection():
    ot = ByteTrack()
    ot.detect()


def run_tracking(
    output_path=None,
    show=False,
    device=None,
    video_path=None,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    image_scale=1,
    save_pred_csv=None,
    save_eval_csv=None,
    save_trackeval_txt=None,
    max_frames=None,
    write_video=True,
    save_csv=True,
    tracker_yaml=None,
    model_path=None,
    stitch_ids=True,
    stitch_max_gap=DEFAULT_STITCH_MAX_GAP,
    stitch_iou=DEFAULT_STITCH_IOU,
    stitch_center_dist=DEFAULT_STITCH_CENTER_DIST,
):
    ot = ByteTrack(device=device, video_path=video_path, tracker_yaml=tracker_yaml, model_path=model_path)
    return ot.track(
        output_path=output_path,
        show=show,
        conf=conf,
        iou=iou,
        image_scale=image_scale,
        save_pred_csv=save_pred_csv,
        save_eval_csv=save_eval_csv,
        save_trackeval_txt=save_trackeval_txt,
        max_frames=max_frames,
        write_video=write_video,
        save_csv=save_csv,
        stitch_ids=stitch_ids,
        stitch_max_gap=stitch_max_gap,
        stitch_iou=stitch_iou,
        stitch_center_dist=stitch_center_dist,
    )


def find_video_files(input_dir):
    input_dir = Path(input_dir)
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    )


def run_tracking_folder(
    input_dir=DEFAULT_INPUT_DIR,
    video_output_dir=DEFAULT_VIDEO_OUTPUT_DIR,
    trackeval_tracker_dir=DEFAULT_TRACKEVAL_TRACKER_DIR,
    show=False,
    device=None,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    image_scale=1,
    max_frames=None,
    write_video=True,
    save_csv=True,
    tracker_yaml=None,
    model_path=None,
    stitch_ids=True,
    stitch_max_gap=DEFAULT_STITCH_MAX_GAP,
    stitch_iou=DEFAULT_STITCH_IOU,
    stitch_center_dist=DEFAULT_STITCH_CENTER_DIST,
):
    input_dir = Path(input_dir)
    video_output_dir = Path(video_output_dir)
    trackeval_tracker_dir = Path(trackeval_tracker_dir)
    csv_output_dir = video_output_dir / "csv"
    video_files = find_video_files(input_dir)

    if not video_files:
        raise FileNotFoundError(f"No video files found in {input_dir}")

    print(f"Found {len(video_files)} videos in {input_dir}")
    for index, video_path in enumerate(video_files, start=1):
        normalized_sequence = normalize_sequence_name(video_path.stem)
        output_path = video_output_dir / f"{video_path.stem}.mp4"
        save_pred_csv = csv_output_dir / f"{normalized_sequence}_prediction.csv" if save_csv else None
        save_eval_csv = csv_output_dir / f"{normalized_sequence}_evaluation.csv" if save_csv else None
        save_trackeval_txt = trackeval_tracker_dir / f"{normalized_sequence}.txt"
        print(f"[{index}/{len(video_files)}] Tracking {video_path.name}")
        run_tracking(
            output_path=output_path,
            show=show,
            device=device,
            video_path=video_path,
            conf=conf,
            iou=iou,
            image_scale=image_scale,
            save_pred_csv=save_pred_csv,
            save_eval_csv=save_eval_csv,
            save_trackeval_txt=save_trackeval_txt,
            max_frames=max_frames,
            write_video=write_video,
            save_csv=save_csv,
            tracker_yaml=tracker_yaml,
            model_path=model_path,
            stitch_ids=stitch_ids,
            stitch_max_gap=stitch_max_gap,
            stitch_iou=stitch_iou,
            stitch_center_dist=stitch_center_dist,
        )

    if write_video:
        print(f"Saved annotated videos to {video_output_dir}")
    else:
        print("Annotated video output disabled")
    if save_csv:
        print(f"Saved CSV files to {csv_output_dir}")
    else:
        print("CSV output disabled")
    print(f"Saved TrackEval TXT files to {trackeval_tracker_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack and save an annotated video.")
    parser.add_argument("--single", action="store_true", help="Track only one video instead of the whole input folder.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Input folder for batch tracking.")
    parser.add_argument("--input-video", default=None, help="Optional single input video path.")
    parser.add_argument("--video-output-dir", default=DEFAULT_VIDEO_OUTPUT_DIR, help="Annotated video output folder for folder tracking.")
    parser.add_argument("--trackeval-tracker-dir", default=DEFAULT_TRACKEVAL_TRACKER_DIR, help="TrackEval tracker TXT output folder.")
    parser.add_argument("--output", "-o", default="outputs/videos/bytetrack.mp4", help="Output video path.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--model", default="models/yolov11n_last.pt", help="YOLO weights path.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Minimum detection confidence passed into tracker.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--tracker-yaml", default=None, help="Optional custom ByteTrack YAML config path.")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor divisor. 2 means half size.")
    parser.add_argument("--save-pred-csv", default=None, help="Optional path to save x1,y1,x2,y2 prediction CSV.")
    parser.add_argument("--save-eval-csv", default=None, help="Optional path to save x,y,w,h CSV for MOTA evaluation.")
    parser.add_argument("--save-trackeval-txt", default="outputs/trackeval/bytetrack.txt", help="Path to save MOTChallenge/TrackEval TXT.")
    parser.add_argument("--no-video", action="store_true", help="Disable annotated video output.")
    parser.add_argument("--no-csv", action="store_true", help="Disable prediction/evaluation CSV output.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--no-stitch-ids", action="store_true", help="Disable stable ID stitching after ByteTrack.")
    parser.add_argument("--stitch-max-gap", type=int, default=DEFAULT_STITCH_MAX_GAP, help="Max missing frames for stable ID stitching.")
    parser.add_argument("--stitch-iou", type=float, default=DEFAULT_STITCH_IOU, help="Minimum IoU for stable ID stitching.")
    parser.add_argument("--stitch-center-dist", type=float, default=DEFAULT_STITCH_CENTER_DIST, help="Max center distance ratio for stable ID stitching.")
    parser.add_argument("--show", action="store_true", help="Show the video while processing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.single:
        run_tracking(
            output_path=args.output,
            show=args.show,
            device=args.device,
            video_path=args.input_video,
            conf=args.conf,
            iou=args.iou,
            image_scale=args.scale,
            save_pred_csv=args.save_pred_csv,
            save_eval_csv=args.save_eval_csv,
            save_trackeval_txt=args.save_trackeval_txt,
            max_frames=args.max_frames,
            write_video=not args.no_video,
            save_csv=not args.no_csv,
            tracker_yaml=args.tracker_yaml,
            model_path=args.model,
            stitch_ids=not args.no_stitch_ids,
            stitch_max_gap=args.stitch_max_gap,
            stitch_iou=args.stitch_iou,
            stitch_center_dist=args.stitch_center_dist,
        )
    else:
        run_tracking_folder(
            input_dir=args.input_dir,
            video_output_dir=args.video_output_dir,
            trackeval_tracker_dir=args.trackeval_tracker_dir,
            show=args.show,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            image_scale=args.scale,
            max_frames=args.max_frames,
            write_video=not args.no_video,
            save_csv=not args.no_csv,
            tracker_yaml=args.tracker_yaml,
            model_path=args.model,
            stitch_ids=not args.no_stitch_ids,
            stitch_max_gap=args.stitch_max_gap,
            stitch_iou=args.stitch_iou,
            stitch_center_dist=args.stitch_center_dist,
        )
