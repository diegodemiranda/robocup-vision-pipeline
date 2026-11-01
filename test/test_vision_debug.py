import numpy as np

from Webots_YOLOv8.detections import Detection, RejectionReason, Verdict, YoloClass
from pipeline import PipelineResult
from Webots_YOLOv8.vision_debug import (
    ACCEPTED_COLOR,
    IGNORED_COLOR,
    REJECTED_COLOR,
    draw_debug_image,
)

FRAME = np.zeros((200, 300, 3), np.uint8)
NO_FIELD = PipelineResult()   # sem campo: nenhuma tintura de fundo, só as caixas


def verdict(rejection, center_x):
    detection = Detection(YoloClass.BALL, 0.9, center_x, 100.0, 40.0, 40.0)
    return Verdict(detection, rejection)


def left_edge_pixel(image, center_x):
    return tuple(int(v) for v in image[100, int(center_x - 20)])


def test_accepted_rejected_and_ignored_boxes_have_distinct_colors():
    verdicts = [verdict(None, 50.0), verdict(RejectionReason.OUTSIDE_FIELD, 150.0),
                verdict(RejectionReason.IGNORED_CLASS, 250.0)]
    image = draw_debug_image(FRAME, NO_FIELD, verdicts)

    assert left_edge_pixel(image, 50.0) == ACCEPTED_COLOR
    assert left_edge_pixel(image, 150.0) == REJECTED_COLOR
    assert left_edge_pixel(image, 250.0) == IGNORED_COLOR


def test_input_frame_is_not_modified_and_shape_is_kept():
    original = FRAME.copy()
    image = draw_debug_image(FRAME, NO_FIELD, [verdict(None, 50.0)])
    assert image.shape == FRAME.shape
    assert np.array_equal(FRAME, original)


def test_no_detections_still_produces_an_image():
    assert draw_debug_image(FRAME, NO_FIELD, []).shape == FRAME.shape
