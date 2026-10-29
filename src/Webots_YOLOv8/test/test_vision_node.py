"""
Teste de integração do nó de visão com rclpy real.

Requer o pacote instalado e carregado (`source install/setup.bash`); sem isso,
o módulo é pulado.
"""

import importlib.util
import os

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('soccer_vision_2d_msgs.msg')

from ament_index_python.packages import (  # noqa: E402
    PackageNotFoundError,
    get_package_share_directory,
)
from rclpy.parameter import Parameter  # noqa: E402

import ros_harness  # noqa: E402

from scene import CROSSING, GOALPOST_BOX, ROBOT_BOX, make_scene, xyxy_to_xywh  # noqa: E402
from Webots_YOLOv8.detections import YoloClass  # noqa: E402

try:
    PACKAGE_SHARE = get_package_share_directory('Webots_YOLOv8')
except PackageNotFoundError:
    pytest.skip('Webots_YOLOv8 não está instalado (rode colcon build e o setup.bash)',
                allow_module_level=True)

CONFIG_FILE = os.path.join(PACKAGE_SHARE, 'config', 'vision_pipeline.yaml')


def detections_for_scene():
    ball_on_field_foot = (500, 600)
    return [
        (YoloClass.BALL, 0.9, [ball_on_field_foot[0], ball_on_field_foot[1] - 15, 30, 30]),
        (YoloClass.BALL, 0.8, [500, 200, 30, 30]),                       # flutuando no céu
        (YoloClass.GOALPOST, 0.9, xyxy_to_xywh(GOALPOST_BOX)),
        (YoloClass.ROBOT, 0.9, xyxy_to_xywh(ROBOT_BOX)),
        (YoloClass.X_INTERSECTION, 0.7, [*CROSSING, 30, 30]),
        (YoloClass.CROSSBAR, 0.6, [500, 600, 100, 10]),                  # nunca publicado
    ]


# ----------------------------------------------------------------------
# Testes
# ----------------------------------------------------------------------

def test_only_validated_detections_reach_the_topics(make_harness):
    harness = make_harness(scripted=detections_for_scene())
    out = harness.process(make_scene())

    (balls,) = out['balls_in_image']
    assert len(balls.balls) == 1 and balls.balls[0].center.y == 585.0   # bola no céu foi barrada
    assert len(out['goal_posts_in_image'][0].posts) == 1
    assert len(out['robots_in_image'][0].robots) == 1
    (intersection,) = out['markings_in_image'][0].intersections
    assert intersection.num_rays == 4

    assert balls.header.frame_id == 'camera_optical_frame' and balls.header.stamp.sec == 42


def test_segmentation_topics_and_debug_image(make_harness):
    harness = make_harness(scripted=detections_for_scene())
    out = harness.process(make_scene())

    (boundary,) = out['field_boundary_in_image']
    assert len(boundary.points) > 50
    # trave e robô já foram explicados pelo YOLO, então não sobram obstáculos
    assert out['obstacles_in_image'][0].obstacles == []

    (debug,) = out['processed_image_topic']
    assert (debug.height, debug.width) == make_scene().shape[:2]


def test_unexplained_obstacle_is_published_when_yolo_misses_it(make_harness):
    harness = make_harness(scripted=[])   # YOLO não viu a trave nem o robô
    out = harness.process(make_scene())
    assert len(out['obstacles_in_image'][0].obstacles) == 2


def test_without_field_arrays_are_empty_and_boundary_is_not_published(make_harness):
    harness = make_harness(scripted=detections_for_scene())
    out = harness.process(make_scene(with_field=False))

    assert out['balls_in_image'][0].balls == []
    assert out['robots_in_image'][0].robots == []
    assert out['field_boundary_in_image'] == []   # fronteira vazia derrubaria o soccer_ipm


def test_validation_can_be_turned_off_at_runtime(make_harness):
    harness = make_harness(scripted=detections_for_scene())
    assert len(harness.process(make_scene())['balls_in_image'][0].balls) == 1

    result = harness.set_parameter('validation.enabled', Parameter.Type.BOOL, False)
    assert result.successful
    assert len(harness.process(make_scene())['balls_in_image'][0].balls) == 2


def test_invalid_parameters_are_rejected_and_old_settings_are_kept(make_harness):
    harness = make_harness()
    before = harness.vision.settings

    bad_hue = harness.set_parameter('hsv.field_min', Parameter.Type.INTEGER_ARRAY, [200, 0, 0])
    bad_ratio = harness.set_parameter('validation.min_marking_ratio', Parameter.Type.DOUBLE, 1.5)
    assert not bad_hue.successful and 'hsv.field_min' in bad_hue.reason
    assert not bad_ratio.successful
    assert harness.vision.settings is before

    good = harness.set_parameter('hsv.field_min', Parameter.Type.INTEGER_ARRAY, [48, 100, 70])
    assert good.successful
    assert harness.vision.settings.color_detector.field_color.h_min == 48
    assert harness.vision.pipeline.color_detector is harness.vision.settings.color_detector


def test_shipped_yaml_config_is_accepted_by_the_node(make_harness):
    harness = make_harness(ros_args=['-r', '__node:=vision_node', '--params-file', CONFIG_FILE])
    settings = harness.vision.settings
    assert settings.pipeline.process_width == 540
    assert settings.validation.footpoint_margin_px == 12
    assert settings.publish_segmentation is True


def test_launch_file_builds_with_the_installed_config():
    spec = importlib.util.spec_from_file_location(
        'vision_launch', os.path.join(PACKAGE_SHARE, 'launch', 'vision.launch.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    description = module.generate_launch_description()
    assert len(description.entities) == 4   # 2 argumentos + nó de visão + soccer_ipm
    assert os.path.isfile(CONFIG_FILE)


def test_model_file_parameter_selects_the_weights(make_harness):
    harness = make_harness(ros_args=['-p', 'model_file:=best.pt'])
    assert harness.vision.model.path.endswith(os.path.join('modelo', 'best.pt'))


def test_missing_model_file_fails_at_startup_with_a_clear_message(make_harness):
    with pytest.raises(FileNotFoundError, match='best.pt'):   # lista os modelos disponíveis
        make_harness(ros_args=['-p', 'model_file:=nao_existe.pt'])


@pytest.mark.parametrize('name, type_, value', [
    ('model_file', Parameter.Type.STRING, 'outro.pt'),
    ('strict_class_check', Parameter.Type.BOOL, False),
])
def test_startup_only_parameters_cannot_change_at_runtime(make_harness, name, type_, value):
    harness = make_harness()
    result = harness.set_parameter(name, type_, value)
    assert not result.successful and 'inicialização' in result.reason


ROBOT_MODEL_CLASSES = {   # esquema do connecting_and_showing.py (robô real)
    0: 'ball', 1: 'robot', 2: 'right_goal', 3: 'left_goal',
    4: 'L-Intersection', 5: 'T-Intersection', 6: 'X-Intersection', 7: 'center',
}


def test_model_with_another_class_scheme_is_refused_at_startup(make_harness, monkeypatch):
    monkeypatch.setattr(ros_harness.FakeYOLO, 'names', ROBOT_MODEL_CLASSES)
    with pytest.raises(ValueError, match='goalpost'):
        make_harness()


def test_class_check_can_be_downgraded_to_a_warning(make_harness, monkeypatch):
    monkeypatch.setattr(ros_harness.FakeYOLO, 'names', ROBOT_MODEL_CLASSES)
    harness = make_harness(ros_args=['-p', 'strict_class_check:=false'])
    assert len(harness.vision.model.names) == 8
