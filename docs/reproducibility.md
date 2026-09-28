# Reproducibility

## 1. Environment

Use Python 3.10+ and install:

```bash
pip install -r requirements.txt
pip install -e .
```

Record the Python, PyTorch, CUDA, and Ultralytics versions used for a reported experiment.

## 2. Dataset

Record:

- UA-DETRAC version/source
- train/validation/test split
- YOLO conversion procedure
- image resolution
- class mapping

## 3. Detector

Record:

- model architecture/checkpoint
- training epochs
- batch size
- image size
- confidence threshold
- NMS IoU threshold
- device

## 4. Tracker

For DeepSORT record:

- `max_age`
- `n_init`
- `max_cosine_distance`
- Re-ID embedder

For ByteTrack record:

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- whether stable ID stitching was enabled

## 5. Evaluation

Export results in MOTChallenge-compatible TXT format and evaluate using a fixed evaluation configuration.

Do not mix outputs from different detector checkpoints or tracker settings in the same comparison table.
