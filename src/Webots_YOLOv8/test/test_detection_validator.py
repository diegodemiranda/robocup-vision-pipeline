import pytest

from scene import (
    CROSSING,
    GOALPOST_BOX,
    HORIZON,
    ROBOT_BOX,
    make_scene,
    xyxy_to_xywh,
)
from Webots_YOLOv8.detection_validator import DetectionValidator, FieldRegion, ValidationConfig
from Webots_YOLOv8.detections import Detection, RejectionReason, YoloClass


def detection(yolo_class, box_xywh):
    return Detection(yolo_class, 0.9, *box_xywh)


def ball_with_foot_at(x, foot_y, size=30.0):
    return detection(YoloClass.BALL, [x, foot_y - size / 2.0, size, size])


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def validator(color_detector):
    return DetectionValidator(color_detector, ValidationConfig())


def reasons(verdicts):
    return [verdict.rejection for verdict in verdicts]


def test_objects_on_the_field_are_accepted(validator, segment, scene):
    detections = [
        ball_with_foot_at(500, 600),
        detection(YoloClass.GOALPOST, xyxy_to_xywh(GOALPOST_BOX)),
        detection(YoloClass.ROBOT, xyxy_to_xywh(ROBOT_BOX)),
    ]
    verdicts = validator.validate(detections, segment(scene), scene)
    assert reasons(verdicts) == [None, None, None]


def test_ball_floating_in_the_sky_is_rejected(validator, segment, scene):
    sky_ball = ball_with_foot_at(500, HORIZON - 100)
    verdict, = validator.validate([sky_ball], segment(scene), scene)
    assert verdict.rejection == RejectionReason.OUTSIDE_FIELD


def test_footpoint_margin_tolerates_small_errors_near_the_boundary(color_detector, segment, scene):
    segmentation = segment(scene)
    just_above = ball_with_foot_at(500, HORIZON - 6)

    strict = DetectionValidator(color_detector, ValidationConfig(footpoint_margin_px=0))
    tolerant = DetectionValidator(color_detector, ValidationConfig(footpoint_margin_px=12))

    strict_verdict, = strict.validate([just_above], segmentation, scene)
    tolerant_verdict, = tolerant.validate([just_above], segmentation, scene)
    assert strict_verdict.rejection == RejectionReason.OUTSIDE_FIELD
    assert tolerant_verdict.accepted


def test_intersection_needs_marking_pixels(validator, segment, scene):
    on_crossing = detection(YoloClass.X_INTERSECTION, [*CROSSING, 30.0, 30.0])
    on_grass = detection(YoloClass.L_INTERSECTION, [600.0, 620.0, 30.0, 30.0])
    in_sky = detection(YoloClass.T_INTERSECTION, [600.0, 100.0, 30.0, 30.0])

    verdicts = validator.validate([on_crossing, on_grass, in_sky], segment(scene), scene)
    assert reasons(verdicts) == [None, RejectionReason.NO_MARKING, RejectionReason.OUTSIDE_FIELD]


def test_crossbar_is_never_published(validator, segment, scene):
    crossbar = detection(YoloClass.CROSSBAR, [500.0, 600.0, 100.0, 10.0])
    verdict, = validator.validate([crossbar], segment(scene), scene)
    assert verdict.rejection == RejectionReason.IGNORED_CLASS


def test_nothing_is_valid_when_no_field_is_segmented(validator, segment):
    empty_scene = make_scene(with_field=False)
    verdict, = validator.validate([ball_with_foot_at(500, 600)], segment(empty_scene), empty_scene)
    assert verdict.rejection == RejectionReason.NO_FIELD_DETECTED


def test_disabled_validation_accepts_everything_but_the_crossbar(color_detector, segment):
    empty_scene = make_scene(with_field=False)
    validator = DetectionValidator(color_detector, ValidationConfig(enabled=False))
    detections = [ball_with_foot_at(500, 100), detection(YoloClass.CROSSBAR, [1.0, 1.0, 2.0, 2.0])]
    verdicts = validator.validate(detections, segment(empty_scene), empty_scene)
    assert reasons(verdicts) == [None, RejectionReason.IGNORED_CLASS]


def test_field_region_maps_points_from_the_processing_resolution(segment, scene):
    segmentation = segment(scene)  # máscara em 540 px de largura, imagem em 1080
    assert segmentation.hull_mask.shape[1] == 540

    region = FieldRegion.from_segmentation(segmentation, scene.shape)
    assert region.contains(500, HORIZON + 40)
    assert not region.contains(500, HORIZON - 40)
    assert region.as_frame_mask().shape == scene.shape[:2]


def test_points_outside_the_image_are_clamped_not_crashing(segment, scene):
    region = FieldRegion.from_segmentation(segment(scene), scene.shape)
    assert region.contains(5000, 5000)   # canto inferior direito é campo
    assert not region.contains(-50, -50)
