import pytest

from Webots_YOLOv8.detections import (
    EXPECTED_CLASS_NAMES,
    Detection,
    YoloClass,
    class_name_mismatches,
    detections_from_yolo,
)


def test_footpoint_and_xyxy_from_center_box():
    detection = Detection(
        YoloClass.ROBOT, 0.9, center_x=100.0, center_y=200.0, width=40.0, height=80.0)
    assert detection.footpoint == (100.0, 240.0)  # centro da base
    assert detection.xyxy == (80.0, 160.0, 120.0, 240.0)


def test_detections_from_yolo_keeps_order_and_drops_unknown_classes():
    detections = detections_from_yolo(
        classes=[2.0, 99.0, 0.0],
        scores=[0.8, 0.7, 0.6],
        boxes_xywh=[[10, 20, 30, 40], [1, 2, 3, 4], [5, 6, 7, 8]],
    )
    assert [d.yolo_class for d in detections] == [YoloClass.ROBOT, YoloClass.BALL]
    assert detections[0].confidence == pytest.approx(0.8)
    assert detections[1].center_x == 5


def test_class_name_mismatches_flags_wrong_model():
    assert class_name_mismatches(dict(EXPECTED_CLASS_NAMES)) == []
    wrong = dict(EXPECTED_CLASS_NAMES)
    wrong[1] = 'post'
    problems = class_name_mismatches(wrong)
    assert len(problems) == 1 and 'goalpost' in problems[0]
