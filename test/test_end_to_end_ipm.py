"""
Ponta a ponta: nó de visão -> soccer_ipm REAL -> posições em metros.

Mostra que o que o nó valida e publica em pixels é aceito pelo IPM e vira a
distância certa no plano do campo. Modelo da câmera do AUREA: 0,54 m de altura
(0,29 + 0,11 + 0,14 do URDF), olhando para frente, 1080x720, FOV de 0,7854 rad.
"""

import math

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('soccer_vision_2d_msgs.msg')
soccer_ipm_module = pytest.importorskip('soccer_ipm.soccer_ipm')
sv3dm = pytest.importorskip('soccer_vision_3d_msgs.msg')

from geometry_msgs.msg import TransformStamped  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster  # noqa: E402

from ros_harness import Harness  # noqa: E402
from scene import GOALPOST_BOX, ROBOT_BOX, make_scene, xyxy_to_xywh  # noqa: E402
from Webots_YOLOv8.detections import YoloClass  # noqa: E402

CAMERA_HEIGHT_M = 0.54
BALL_DIAMETER_M = 0.14
IMAGE_WIDTH, IMAGE_HEIGHT = 1080, 720
FOCAL_PX = (IMAGE_WIDTH / 2) / math.tan(0.7854 / 2)
OPTICAL_ROTATION = (-0.5, 0.5, -0.5, 0.5)  # rpy = (-pi/2, 0, -pi/2) do URDF

RELATIVE_TOPICS = {
    'balls_relative': sv3dm.BallArray,
    'goal_posts_relative': sv3dm.GoalpostArray,
    'field_boundary_relative': sv3dm.FieldBoundary,
    'obstacles_relative': sv3dm.ObstacleArray,
}
IPM_ARGS = [
    '-p', 'output_frame:=base_link',
    '-p', f'balls.ball_diameter:={BALL_DIAMETER_M}',
    '-r', 'camera_info:=/AUREA/camera_optical_frame/camera_info',
]


class IpmHarness(Harness):
    """Harness com o soccer_ipm real, TF estático e CameraInfo."""

    publish_camera_info = True   # False simula o início, antes de a câmera informar seus parâmetros

    def _required_subscribers(self, topic):
        # o debug só tem o probe; os demais também têm o soccer_ipm
        return 1 if topic == 'processed_image_topic' else 2

    def _add_extra_nodes(self):
        self.ipm = soccer_ipm_module.SoccerIPM()
        self.executor.add_node(self.ipm)

        self.relative = {topic: [] for topic in RELATIVE_TOPICS}
        for topic, msg_type in RELATIVE_TOPICS.items():
            self.probe.create_subscription(
                msg_type, topic, lambda msg, t=topic: self.relative[t].append(msg), 10)

        self._static_tf = StaticTransformBroadcaster(self.probe)
        self._static_tf.sendTransform(self._camera_transform())
        self._camera_info_pub = self.probe.create_publisher(
            CameraInfo, '/AUREA/camera_optical_frame/camera_info', 10)
        self.probe.create_timer(0.1, self._publish_camera_info)

    @staticmethod
    def _camera_transform():
        transform = TransformStamped()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'camera_optical_frame'
        transform.transform.translation.z = CAMERA_HEIGHT_M
        (transform.transform.rotation.x, transform.transform.rotation.y,
         transform.transform.rotation.z, transform.transform.rotation.w) = OPTICAL_ROTATION
        return transform

    def _connected(self):
        ipm_has_camera_info = self.ipm.ipm.camera_info_received()
        return super()._connected() and (ipm_has_camera_info or not self.publish_camera_info)

    def _publish_camera_info(self):
        if not self.publish_camera_info:
            return
        info = CameraInfo()
        info.header.frame_id = 'camera_optical_frame'
        info.width, info.height = IMAGE_WIDTH, IMAGE_HEIGHT
        info.k = [FOCAL_PX, 0.0, IMAGE_WIDTH / 2, 0.0, FOCAL_PX, IMAGE_HEIGHT / 2, 0.0, 0.0, 1.0]
        info.p = [FOCAL_PX, 0.0, IMAGE_WIDTH / 2, 0.0,
                  0.0, FOCAL_PX, IMAGE_HEIGHT / 2, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.d = [0.0] * 5
        info.distortion_model = 'plumb_bob'
        self._camera_info_pub.publish(info)

    def process_until_relative(self, frame, topic):
        for messages in self.relative.values():
            messages.clear()
        self.process(frame)
        self._wait_until(lambda: self.relative[topic], f'mensagem em {topic}')
        return self.relative


def expected_ground_distance(pixel_row, height_above_plane):
    """Distância à frente de um ponto que está a `height_above_plane` m do plano da câmera."""
    return (CAMERA_HEIGHT_M - height_above_plane) * FOCAL_PX / (pixel_row - IMAGE_HEIGHT / 2)


def test_validated_ball_becomes_the_right_distance_in_meters(make_harness):
    ball_row = 585.0
    scripted = [
        (YoloClass.BALL, 0.9, [IMAGE_WIDTH / 2, ball_row, 30, 30]),   # no campo, à frente
        (YoloClass.BALL, 0.9, [IMAGE_WIDTH / 2, 150, 30, 30]),        # no céu: barrada
    ]
    harness = make_harness(IPM_ARGS, scripted, harness_class=IpmHarness)

    relative = harness.process_until_relative(make_scene(), 'balls_relative')

    (balls,) = relative['balls_relative']
    assert balls.header.frame_id == 'base_link'
    assert len(balls.balls) == 1   # a bola do céu nem chegou ao IPM
    center = balls.balls[0].center
    expected = expected_ground_distance(ball_row, BALL_DIAMETER_M / 2)
    assert center.x == pytest.approx(expected, rel=0.01)
    assert center.y == pytest.approx(0.0, abs=0.01)   # no eixo da câmera


def test_goalpost_foot_is_projected_to_the_ground(make_harness):
    scripted = [(YoloClass.GOALPOST, 0.9, xyxy_to_xywh(GOALPOST_BOX))]
    harness = make_harness(IPM_ARGS, scripted, harness_class=IpmHarness)

    relative = harness.process_until_relative(make_scene(), 'goal_posts_relative')

    (post,) = relative['goal_posts_relative'][0].posts
    foot_row = GOALPOST_BOX[3]
    expected = expected_ground_distance(foot_row, 0.0)
    assert post.bb.center.position.x == pytest.approx(expected, rel=0.01)


def _ipm_handles_missing_camera_info():
    """True se o soccer_ipm instalado tem o patch de camera_info (veja docs/development.md)."""
    from ipm_library.exceptions import CameraInfoNotSetException
    from soccer_ipm.utils import object_at_bottom_of_image
    try:
        object_at_bottom_of_image(None, 0.0, 0.8)
    except CameraInfoNotSetException:
        return True
    except Exception:   # AttributeError no soccer_ipm sem o patch
        return False
    return True


@pytest.mark.skipif(not _ipm_handles_missing_camera_info(),
                    reason='soccer_ipm sem o patch de camera_info (veja docs/development.md)')
def test_ipm_survives_frames_that_arrive_before_camera_info(make_harness):
    """Regressão: sem CameraInfo, o soccer_ipm levantava AttributeError e caía."""
    scripted = [
        (YoloClass.GOALPOST, 0.9, xyxy_to_xywh(GOALPOST_BOX)),
        (YoloClass.ROBOT, 0.9, xyxy_to_xywh(ROBOT_BOX)),
    ]
    harness = make_harness(IPM_ARGS, scripted, harness_class=IpmHarness)
    harness.publish_camera_info = False
    harness.ipm.ipm.set_camera_info(None)

    harness.process(make_scene())            # uma exceção no IPM estouraria aqui
    assert harness.relative['goal_posts_relative'] == []

    harness.publish_camera_info = True       # a câmera começa a informar seus parâmetros
    harness._wait_until(harness.ipm.ipm.camera_info_received, 'camera_info no IPM')
    relative = harness.process_until_relative(make_scene(), 'goal_posts_relative')
    assert len(relative['goal_posts_relative'][0].posts) == 1


def test_segmentation_topics_are_accepted_by_the_ipm(make_harness):
    harness = make_harness(IPM_ARGS, [], harness_class=IpmHarness)

    relative = harness.process_until_relative(make_scene(), 'field_boundary_relative')

    assert len(relative['field_boundary_relative'][0].points) > 0
    harness.process_until_relative(make_scene(), 'obstacles_relative')  # YOLO não viu: 2 obstáculos
