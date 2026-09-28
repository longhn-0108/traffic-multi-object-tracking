# Data

Large datasets and videos are intentionally excluded from Git.

## UA-DETRAC

The project expects a YOLO-format version of UA-DETRAC for detector training and local video files for tracking.

Recommended local layout:

```text
data/
└── videos/
    ├── MVI_39031.mp4
    ├── MVI_39051.mp4
    └── ...
```

For training data, configure the dataset location in:

```text
configs/ua_detrac.yaml
```

Do not commit the raw UA-DETRAC dataset or large videos to the repository.
