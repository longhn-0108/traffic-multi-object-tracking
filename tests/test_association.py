from mot_tracking.association import StableIdStitcher, box_iou


def test_box_iou_identical_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_box_iou_non_overlapping_boxes():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_stable_id_stitching():
    stitcher = StableIdStitcher(enabled=True, max_gap=3, iou_threshold=0.1)

    first = stitcher.resolve(1, [0, 0, 10, 10], 0, 1)
    second = stitcher.resolve(2, [1, 0, 11, 10], 0, 2)

    assert first == 1
    assert second == 1
    assert stitcher.stitched_fragments == 1
