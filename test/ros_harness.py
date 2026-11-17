"""
Infraestrutura dos testes com ROS real: YOLO falso, nó "probe" e fixture `make_harness`.

O YOLO é substituído por um falso (módulo `ultralytics` simulado), mas o
`running_inference.py`, a segmentação, a validação e as mensagens são os reais.
"""

import importlib
import sys
import time
import types

import rclpy
import soccer_vision_2d_msgs.msg as sv2dm
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from Webots_YOLOv8.detections import EXPECTED_CLASS_NAMES

CAMERA_TOPIC = '/AUREA/camera_optical_frame/image_color'
OUTPUT_TOPICS = {
    'balls_in_image': sv2dm.BallArray,
    'goal_posts_in_image': sv2dm.GoalpostArray,
    'robots_in_image': sv2dm.RobotArray,
    'markings_in_image': sv2dm.MarkingArray,
    'field_boundary_in_image': sv2dm.FieldBoundary,
    'obstacles_in_image': sv2dm.ObstacleArray,
    'processed_image_topic': Image,
}


# ----------------------------------------------------------------------
# YOLO falso: imita a API do ultralytics usada por running_inference.py
# ----------------------------------------------------------------------

class _Values(list):
    def tolist(self):
        return list(self)


class FakeYOLO:
    names = {int(k): v for k, v in EXPECTED_CLASS_NAMES.items()}
    scripted = []   # [(classe, score, [cx, cy, w, h])] devolvido a cada frame

    def __init__(self, path):
        self.path = path

    def predict(self, source, **_kwargs):
        boxes = types.SimpleNamespace(
            cls=_Values(c for c, _, _ in self.scripted),
            conf=_Values(s for _, s, _ in self.scripted),
            xywh=_Values(b for _, _, b in self.scripted),
        )
        return [types.SimpleNamespace(boxes=boxes, plot=lambda: source.copy())]


# ----------------------------------------------------------------------
# Harness: nó de visão + nó "probe" que publica imagens e escuta as saídas
# ----------------------------------------------------------------------

class Harness:
    def __init__(self, ros_args):
        node_module = self._import_node_module()
        rclpy.init(args=['--ros-args', '-p', 'show_window:=false', *ros_args])
        try:
            self.vision = node_module.YoloSimulacao()
        except Exception:
            rclpy.shutdown()  # não deixa o contexto aberto para o próximo teste
            raise
        self.probe = rclpy.create_node('vision_probe')
        self.bridge = CvBridge()
        self.received = {topic: [] for topic in OUTPUT_TOPICS}

        self.image_pub = self.probe.create_publisher(Image, CAMERA_TOPIC, 10)
        for topic, msg_type in OUTPUT_TOPICS.items():
            self.probe.create_subscription(
                msg_type, topic, lambda msg, t=topic: self.received[t].append(msg), 10)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.vision)
        self.executor.add_node(self.probe)
        self._add_extra_nodes()
        self._wait_until(self._connected, 'conexão entre os nós')

    def _add_extra_nodes(self):
        """Gancho para subclasses que precisam de mais nós (ex.: soccer_ipm)."""

    @staticmethod
    def _import_node_module():
        for name in ('Webots_YOLOv8.running_inference', 'Webots_YOLOv8.yolo_simulation'):
            sys.modules.pop(name, None)  # recarrega usando o ultralytics falso
        return importlib.import_module('Webots_YOLOv8.yolo_simulation')

    def _required_subscribers(self, topic):
        """Quantos assinantes um tópico de saída precisa ter antes de publicar o frame."""
        return 1  # o probe; subclasses somam os seus

    def _connected(self):
        if self.image_pub.get_subscription_count() < 1:
            return False
        for publisher in self.vision.publishers:
            topic = publisher.topic_name.lstrip('/')
            if topic in OUTPUT_TOPICS:
                if publisher.get_subscription_count() < self._required_subscribers(topic):
                    return False
        return True

    def _wait_until(self, condition, what, timeout=15.0):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError(f'tempo esgotado esperando: {what}')
            self.executor.spin_once(timeout_sec=0.1)

    def process(self, frame, last_topic='obstacles_in_image'):
        """Publica um frame e devolve as mensagens recebidas em cada tópico."""
        for messages in self.received.values():
            messages.clear()
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.frame_id = 'camera_optical_frame'
        msg.header.stamp.sec = 42
        self.image_pub.publish(msg)

        self._wait_until(lambda: self.received[last_topic], f'mensagem em {last_topic}')
        for _ in range(3):  # deixa as demais mensagens do mesmo frame chegarem
            self.executor.spin_once(timeout_sec=0.1)
        return self.received

    def set_parameter(self, name, type_, value):
        return self.vision.set_parameters([Parameter(name, type_, value)])[0]

    def close(self):
        self.executor.shutdown()
        self.vision.destroy_node()
        self.probe.destroy_node()
        rclpy.shutdown()


def harness_factory(monkeypatch, tmp_path):
    """Gerador usado pela fixture `make_harness` (conftest.py).

    Uso: `make_harness(scripted=[...], ros_args=[...], harness_class=Harness)`.
    O YOLO é falso e lê só o caminho dos pesos; por padrão aponta para um arquivo vazio em
    `tmp_path`, então os testes não dependem de pesos reais. `ros_args` pode sobrescrever.
    """
    fake_module = types.ModuleType('ultralytics')
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, 'ultralytics', fake_module)
    monkeypatch.setenv('ROS_DOMAIN_ID', '87')  # isola do resto da rede

    fake_weights = tmp_path / 'pesos_de_teste.pt'
    fake_weights.write_bytes(b'')
    harnesses = []

    def _make(ros_args=(), scripted=(), harness_class=Harness):
        FakeYOLO.scripted = list(scripted)
        args = ['-p', f'model_file:={fake_weights}', *ros_args]
        harnesses.append(harness_class(args))
        return harnesses[-1]

    yield _make
    for harness in harnesses:
        harness.close()
    FakeYOLO.scripted = []
    for name in ('Webots_YOLOv8.running_inference', 'Webots_YOLOv8.yolo_simulation'):
        sys.modules.pop(name, None)
