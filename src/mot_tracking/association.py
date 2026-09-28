"""Reusable tracking association utilities."""

from __future__ import annotations

import math


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def predict_box(box, velocity, frame_gap):
    return [box[i] + velocity[i] * frame_gap for i in range(4)]


def center_distance_ratio(box_a, box_b) -> float:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    diagonal = math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])
    if diagonal <= 0:
        return float("inf")
    return math.hypot(bx - ax, by - ay) / diagonal


class StableIdStitcher:
    """Optionally reconnect short ID fragments using motion and IoU cues."""

    def __init__(
        self,
        enabled=True,
        max_gap=15,
        iou_threshold=0.25,
        center_dist_threshold=0.6,
    ):
        self.enabled = enabled
        self.max_gap = max_gap
        self.iou_threshold = iou_threshold
        self.center_dist_threshold = center_dist_threshold
        self.raw_to_stable = {}
        self.raw_ids = set()
        self.stable_tracks = {}
        self.stitched_fragments = 0

    def resolve(self, raw_track_id, box, class_id, frame_num, blocked_stable_ids=None):
        raw_track_id = int(raw_track_id)
        class_id = int(class_id)
        box = [float(value) for value in box]
        self.raw_ids.add(raw_track_id)

        if not self.enabled:
            return raw_track_id

        if raw_track_id in self.raw_to_stable:
            stable_id = self.raw_to_stable[raw_track_id]
        else:
            stable_id = self._find_match(
                box, class_id, frame_num, blocked_stable_ids or set()
            )
            if stable_id is None:
                stable_id = raw_track_id
            else:
                self.stitched_fragments += 1
            self.raw_to_stable[raw_track_id] = stable_id

        self._update_track(stable_id, box, class_id, frame_num)
        return stable_id

    def _find_match(self, box, class_id, frame_num, blocked_stable_ids):
        best_stable_id = None
        best_rank = None

        for stable_id, track in self.stable_tracks.items():
            if stable_id in blocked_stable_ids or track["class_id"] != class_id:
                continue

            frame_gap = frame_num - track["last_frame"]
            if frame_gap <= 0 or frame_gap > self.max_gap:
                continue

            candidate_box = predict_box(
                track["last_box"], track["velocity"], frame_gap
            )
            iou_score = box_iou(candidate_box, box)
            center_ratio = center_distance_ratio(candidate_box, box)

            if (
                iou_score < self.iou_threshold
                and center_ratio > self.center_dist_threshold
            ):
                continue

            rank = (
                iou_score >= self.iou_threshold,
                iou_score,
                -center_ratio,
                -frame_gap,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_stable_id = stable_id

        return best_stable_id

    def _update_track(self, stable_id, box, class_id, frame_num):
        previous = self.stable_tracks.get(stable_id)
        velocity = [0.0, 0.0, 0.0, 0.0]

        if previous is not None:
            frame_gap = frame_num - previous["last_frame"]
            if frame_gap > 0:
                velocity = [
                    (box[i] - previous["last_box"][i]) / frame_gap
                    for i in range(4)
                ]
            else:
                velocity = previous["velocity"]

        self.stable_tracks[stable_id] = {
            "class_id": class_id,
            "last_box": box,
            "last_frame": frame_num,
            "velocity": velocity,
        }

    @property
    def raw_count(self):
        return len(self.raw_ids)

    @property
    def stable_count(self):
        if not self.enabled:
            return len(self.raw_ids)
        return len(set(self.raw_to_stable.values()))
