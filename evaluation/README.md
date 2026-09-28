# Evaluation

This directory contains project-specific helpers for preparing tracking outputs.

## Confidence filtering

```bash
python evaluation/filter_confidence.py \
  --input outputs/trackeval/bytetrack \
  --output outputs/trackeval/bytetrack_filtered \
  --thresh 0.5
```

## Ignore-region filtering

```bash
python evaluation/filter_ignore_regions.py \
  --trk outputs/trackeval/bytetrack \
  --ignore /path/to/ignore_regions \
  --out outputs/trackeval/bytetrack_filtered
```

## TrackEval

The upstream TrackEval source is included under:

```text
third_party/TrackEval/
```

The original benchmark data and generated results are deliberately excluded.

For an experiment, first generate MOTChallenge-compatible tracker TXT files, then configure the corresponding TrackEval dataset paths. See `third_party/TrackEval/Readme.md` for the upstream evaluation interface.

**Important:** UA-DETRAC is not one of TrackEval's standard benchmark classes. If you adapt TrackEval for UA-DETRAC, document the dataset adapter and class mapping before reporting metrics.
