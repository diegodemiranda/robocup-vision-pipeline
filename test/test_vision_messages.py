"""Mensagens ROS reais: o rclpy valida tipos (ex.: int em campo double levanta AssertionError)."""

import pytest

pytest.importorskip('rclpy')
sv2dm = pytest.importorskip('soccer_vision_2d_msgs.msg')

from std_msgs.msg import Header  # noqa: E402

from Webots_YOLOv8.detections import Detection, YoloClass  # noqa: E402
from pipeline import Obstacle  # noqa: E402
from Webots_YOLOv8.vision_messages import (  # noqa: E402
    build_detection_messages,
    build_field_boundary_message,
    build_obstacle_array_message,
)


def make_header():
    header = Header()
    header.frame_id = 'camera_optical_frame'
    header.stamp.sec = 12
    return header


def make_detection(yolo_class, cx=100, cy=200, w=40, h=60, score=0.8):
    # ints de propósito: o conversor precisa transformá-los em float
    return Detection(yolo_class, score, cx, cy, w, h)


def test_each_class_goes_to_its_topic_message():
    detections = [
        make_detection(YoloClass.BALL),
        make_detection(YoloClass.GOALPOST),
        make_detection(YoloClass.ROBOT),
        make_detection(YoloClass.L_INTERSECTION),
        make_detection(YoloClass.T_INTERSECTION),
        make_detection(YoloClass.X_INTERSECTION),
        make_detection(YoloClass.CROSSBAR),
    ]
    messages = build_detection_messages(make_header(), detections)

    assert len(messages.balls.balls) == 1
    assert len(messages.goalposts.posts) == 1
    assert len(messages.robots.robots) == 1
    assert [i.num_rays for i in messages.markings.intersections] == [2, 3, 4]
    for array in (messages.balls, messages.goalposts, messages.robots, messages.markings):
        assert array.header.frame_id == 'camera_optical_frame'


def test_ball_geometry_matches_what_the_ipm_reads():
    ball = build_detection_messages(make_header(), [make_detection(YoloClass.BALL)]).balls.balls[0]
    assert (ball.center.x, ball.center.y) == (100.0, 200.0)
    assert (ball.bb.center.position.x, ball.bb.size_y) == (100.0, 60.0)
    assert ball.confidence.confidence == pytest.approx(0.8)


def test_goalpost_and_robot_bounding_boxes():
    messages = build_detection_messages(
        make_header(), [make_detection(YoloClass.GOALPOST), make_detection(YoloClass.ROBOT)])
    for bb in (messages.goalposts.posts[0].bb, messages.robots.robots[0].bb):
        assert (bb.center.position.x, bb.center.position.y) == (100.0, 200.0)
        assert (bb.size_x, bb.size_y) == (40.0, 60.0)


def test_empty_detections_still_produce_valid_empty_messages():
    messages = build_detection_messages(make_header(), [])
    assert messages.balls.balls == [] and messages.markings.intersections == []


def test_field_boundary_is_none_when_empty_and_float_points_otherwise():
    assert build_field_boundary_message(make_header(), []) is None
    message = build_field_boundary_message(make_header(), [(0, 360), (4, 361)])
    assert [(p.x, p.y) for p in message.points] == [(0.0, 360.0), (4.0, 361.0)]


def test_obstacle_message_uses_box_center():
    obstacle = Obstacle(x=10, y=20, width=30, height=40, color_class='white')
    message = build_obstacle_array_message(make_header(), [obstacle])
    bb = message.obstacles[0].bb
    assert (bb.center.position.x, bb.center.position.y) == (25.0, 40.0)
    assert (bb.size_x, bb.size_y) == (30.0, 40.0)
    assert build_obstacle_array_message(make_header(), []).obstacles == []
