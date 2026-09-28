# Model Weights

Model checkpoints are intentionally excluded from Git.

Place the detector checkpoint used for tracking here, for example:

```text
models/
└── yolov11n_last.pt
```

Then run:

```bash
python scripts/run_deepsort.py \
  --video data/videos/MVI_39031.mp4 \
  --model models/yolov11n_last.pt
```

For reproducible experiments, record the checkpoint filename, training configuration, and source commit in your experiment notes.
