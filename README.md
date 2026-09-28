# Traffic Multi-Object Tracking

A reproducible computer-vision pipeline for **vehicle detection and multi-object tracking (MOT)** on traffic videos.

The project combines a YOLO-based detector with multiple tracking approaches, primarily **DeepSORT** and **ByteTrack**, and exports results in MOTChallenge/TrackEval-compatible formats for downstream evaluation.

## Project Overview

```text
Traffic video
     │
     ▼
YOLO vehicle detector
     │
     ├───────────────┐
     ▼               ▼
 DeepSORT        ByteTrack
     │               │
     └───────┬───────┘
             ▼
   Track IDs + bounding boxes
             │
             ├── Annotated video
             ├── Prediction CSV
             └── MOTChallenge TXT
                    │
                    ▼
              Track evaluation
```

### Methods

| Component | Role |
|---|---|
| YOLO | Vehicle detection |
| DeepSORT | Appearance + motion based association |
| ByteTrack | High/low-confidence detection association |
| Stable ID Stitching | Optional post-processing for short ByteTrack ID fragments |
| TrackEval | Standard MOT metrics/evaluation framework |

## Repository Structure

```text
traffic-multi-object-tracking/
├── src/
│   └── mot_tracking/
│       ├── association.py
│       ├── bytetrack_tracker.py
│       └── deepsort_tracker.py
├── scripts/
│   ├── run_bytetrack.py
│   ├── run_deepsort.py
│   ├── run_detection.py
│   └── train_detector.py
├── configs/
│   ├── bytetrack.yaml
│   └── ua_detrac.yaml
├── evaluation/
│   ├── filter_confidence.py
│   ├── filter_ignore_regions.py
│   └── results/
├── notebooks/
│   ├── metric.ipynb
│   ├── sort.ipynb
│   ├── yolo_detection_ua_detrac.ipynb
│   └── merge_image.ipynb
├── data/
│   └── README.md
├── models/
│   └── README.md
├── third_party/
│   └── TrackEval/
├── tests/
├── outputs/
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

Large datasets, videos, generated outputs, and model checkpoints are intentionally excluded from Git. See the reproducibility guide for how to provide them locally.

## Installation

Python **3.10+** is recommended.

```bash
git clone https://github.com/longhn-0108/traffic-multi-object-tracking.git
cd traffic-multi-object-tracking

python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset

The experiments use **UA-DETRAC** traffic videos and annotations.

The dataset itself is not included in this repository because of its size and licensing/distribution considerations.

For a YOLO-format local dataset, update:

```text
configs/ua_detrac.yaml
```

Example:

```yaml
path: /path/to/ua-detrac-yolo
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: vehicle
```

See [`data/README.md`](data/README.md).

## Model Weights

Model checkpoints are deliberately not committed to Git.

Place trained/pretrained weights under:

```text
models/
```

For example:

```text
models/
└── yolov11n_last.pt
```

See [`models/README.md`](models/README.md).

## Running DeepSORT

Run DeepSORT on one video:

```bash
python scripts/run_deepsort.py   --video data/videos/MVI_39031.mp4   --model models/yolov11n_last.pt   --save-video
```

The default outputs are:

```text
outputs/
├── videos/
│   └── deepsort.mp4
└── trackeval/
    └── deepsort.txt
```

Useful options:

```bash
python scripts/run_deepsort.py --help
```

## Running ByteTrack

Single video:

```bash
python scripts/run_bytetrack.py   --single   --input-video data/videos/MVI_39031.mp4   --model models/yolov11n_last.pt   --output outputs/videos/bytetrack.mp4
```

Run a directory of videos:

```bash
python scripts/run_bytetrack.py   --input-dir data/videos   --model models/yolov11n_last.pt
```

For a fast smoke test:

```bash
python scripts/run_bytetrack.py   --single   --input-video data/videos/MVI_39031.mp4   --model models/yolov11n_last.pt   --max-frames 100   --no-video   --no-csv
```

### ByteTrack configuration

The tracker configuration is stored in:

```text
configs/bytetrack.yaml
```

The project also contains an optional **stable ID stitching** step. This reconnects short fragments using predicted box position, IoU, and center-distance constraints.

Disable it with:

```bash
--no-stitch-ids
```

## Training the Detector

Train a YOLO detector on a YOLO-format UA-DETRAC dataset:

```bash
python scripts/train_detector.py   --data configs/ua_detrac.yaml   --model yolo26n.pt   --epochs 80   --batch 16   --imgsz 640
```

The script does not assume a specific local dataset path or GPU.

## Detection Only

```bash
python scripts/run_detection.py   --model models/yolov11n_last.pt   --source data/videos/MVI_39031.mp4   --conf 0.25   --save
```

## Output Formats

The tracking pipelines can produce:

### Annotated video

Human-readable visualization of:

- bounding boxes
- confidence
- track IDs

### Prediction CSV

Contains:

```text
sequence
frame_num
track_id
x1
y1
x2
y2
confidence
class_id
```

### Evaluation CSV

Contains:

```text
sequence
frame_num
track_id
x
y
w
h
confidence
class_id
```

### MOTChallenge / TrackEval TXT

The standard output contains frame-level tracking records:

```text
frame,id,x,y,w,h,confidence,class,visibility,...
```

## Evaluation

The repository vendors the open-source **TrackEval** implementation under:

```text
third_party/TrackEval/
```

Only the evaluation framework is included; the original benchmark data and generated tracker outputs are not.

See:

- [`evaluation/README.md`](evaluation/README.md)
- [`third_party/TrackEval/Readme.md`](third_party/TrackEval/Readme.md)

For fair experiments, report the exact:

- detector checkpoint
- confidence threshold
- tracker configuration
- dataset split
- image scaling
- ID-stitching configuration
- evaluation metrics

Do not compare runs with different experimental settings without documenting the change.

## Reproducibility

The repository intentionally avoids machine-specific paths such as:

```text
C:\Users\...
E:\CVProject\...
/path/to/dataset
```

All runtime paths are supplied through CLI arguments or repository-relative defaults.

See [`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations

- Tracking quality depends strongly on the detector's recall and localization quality.
- DeepSORT uses the `deep-sort-realtime` implementation and its pretrained appearance embedder by default.
- ByteTrack's optional ID stitching is a project-specific post-processing step and should be evaluated separately from the native tracker.
- Dataset videos and model weights are external dependencies.
- Benchmark numbers should only be reported after rerunning the evaluation with a documented experimental configuration.

## Credits

This project builds on:

- Ultralytics YOLO
- `deep-sort-realtime`
- ByteTrack
- TrackEval
- UA-DETRAC

See the corresponding licenses and citations in [`third_party/`](third_party/) and the project documentation.


## References

- Wojke, Bewley, Paulus. *Simple Online and Realtime Tracking with a Deep Association Metric* — DeepSORT.
- Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* — ByteTrack.
- Luiten et al. *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking* — HOTA.
- Wen et al. *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking* — UA-DETRAC.

## License

The original project code in this repository is released under the MIT License. Third-party components retain their original licenses.
