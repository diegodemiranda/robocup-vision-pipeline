#!/usr/bin/env python3
"""
yolo_simulation.py

Nó de visão do robô AUREA no Webots. Fluxo de cada frame da câmera:

    imagem -> YOLOv8 (detecções) ----------------------+
          \\-> pipeline de segmentação (campo, obstáculos) -> validação -> tópicos ROS 2

Uma detecção só é publicada se estiver dentro da área segmentada (veja
`detection_validator`). Os tópicos `*_in_image` são consumidos pelo
`soccer_ipm`, que projeta os pixels em metros para movimento e behavior.
"""

import os
import traceback
from typing import List, Sequence

import cv2
import rclpy
import soccer_vision_2d_msgs.msg as sv2dm
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Image as ROS_Image
from std_msgs.msg import Header

import Webots_YOLOv8.running_inference as ri
from Webots_YOLOv8.detection_validator import DetectionValidator
from Webots_YOLOv8.detections import (
    FIELD_OBJECT_CLASSES,
    Detection,
    Verdict,
    class_name_mismatches,
    detections_from_yolo,
)
from Webots_YOLOv8.pipeline import PipelineResult, VisionPipeline, fuse_obstacles_with_detections
from Webots_YOLOv8.vision_debug import draw_debug_image
from Webots_YOLOv8.vision_messages import (
    build_detection_messages,
    build_field_boundary_message,
    build_obstacle_array_message,
)
from Webots_YOLOv8.vision_parameters import (
    VisionSettings,
    declare_parameters,
    default_hsv_ranges,
    read_settings,
)

CAMERA_TOPIC = '/AUREA/camera_optical_frame/image_color'
WINDOW_NAME = 'detection window'
LOG_PERIOD_SEC = 5.0
STARTUP_ONLY_PARAMETERS = ('model_file', 'strict_class_check')


class YoloSimulacao(Node):

    def __init__(self):
        super().__init__('teste_yolo_sim')
        self.get_logger().info('>> VISÃO: YOLOv8 + segmentação + validação <<')

        package_share = get_package_share_directory('Webots_YOLOv8')
        green_csv = os.path.join(package_share, 'recursos', 'green_pixels.csv')

        # 1. IA: pesos em modelo/, escolhidos na partida (veja config/vision_pipeline.yaml)
        self.declare_parameter('model_file', ri.DEFAULT_MODEL_FILE)
        self.declare_parameter('strict_class_check', True)
        self.startup_values = {
            name: self.get_parameter(name).value for name in STARTUP_ONLY_PARAMETERS}
        self.model_file = self.startup_values['model_file']
        self.model = ri.load_model(self.model_file)
        self._check_model_classes(strict=self.startup_values['strict_class_check'])

        # 2. Ponte CV <-> ROS
        self.bridge = CvBridge()

        # 3. Parâmetros, segmentação e validação
        declare_parameters(self, default_hsv_ranges(green_csv))
        self.settings: VisionSettings = read_settings(self)
        self.pipeline = VisionPipeline(self.settings.color_detector, self.settings.pipeline)
        self.validator = DetectionValidator(self.settings.color_detector, self.settings.validation)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # 4. Publishers consumidos pelo soccer_ipm
        self.pub_ball = self.create_publisher(sv2dm.BallArray, 'balls_in_image', 1)
        self.pub_goal = self.create_publisher(sv2dm.GoalpostArray, 'goal_posts_in_image', 1)
        self.pub_robot = self.create_publisher(sv2dm.RobotArray, 'robots_in_image', 1)
        self.pub_inter = self.create_publisher(sv2dm.MarkingArray, 'markings_in_image', 1)
        self.pub_boundary = self.create_publisher(
            sv2dm.FieldBoundary, 'field_boundary_in_image', 1)
        self.pub_obstacles = self.create_publisher(
            sv2dm.ObstacleArray, 'obstacles_in_image', 1)

        # 5. Debug e entrada da câmera
        self.debug_pub = self.create_publisher(ROS_Image, 'processed_image_topic', 10)
        self.camera_subscriber = self.create_subscription(
            ROS_Image, CAMERA_TOPIC, self.image_callback, 10)

    # ------------------------------------------------------------------
    # Pipeline de um frame
    # ------------------------------------------------------------------

    def image_callback(self, image_msg: ROS_Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            detections = self._detect(frame)
            segmentation = self.pipeline.run(frame)
            verdicts = self.validator.validate(detections, segmentation, frame)

            self._publish(image_msg.header, verdicts, segmentation)
            self._publish_debug(image_msg.header, frame, segmentation, verdicts)
        except Exception:  # um frame ruim não pode derrubar o nó
            self.get_logger().error(
                f'Falha ao processar o frame:\n{traceback.format_exc()}',
                throttle_duration_sec=LOG_PERIOD_SEC)

    def _detect(self, frame) -> List[Detection]:
        classes, scores, boxes, _ = ri.detect_model(self.model, frame, draw=False)
        return detections_from_yolo(classes, scores, boxes)

    def _publish(
        self, header: Header, verdicts: Sequence[Verdict], segmentation: PipelineResult,
    ) -> None:
        accepted = [verdict.detection for verdict in verdicts if verdict.accepted]

        messages = build_detection_messages(header, accepted)
        self.pub_ball.publish(messages.balls)
        self.pub_goal.publish(messages.goalposts)
        self.pub_robot.publish(messages.robots)
        self.pub_inter.publish(messages.markings)

        if self.settings.publish_segmentation:
            self._publish_segmentation(header, segmentation, accepted)

        self.get_logger().info(
            f'YOLO: {len(verdicts)} | válidas: {len(accepted)} | '
            f'campo: {"sim" if segmentation.field_detected else "não"}',
            throttle_duration_sec=LOG_PERIOD_SEC)

    def _publish_segmentation(
        self, header: Header, segmentation: PipelineResult, accepted: Sequence[Detection],
    ) -> None:
        boundary = build_field_boundary_message(header, segmentation.field_boundary)
        if boundary is not None:  # fronteira vazia derruba o soccer_ipm
            self.pub_boundary.publish(boundary)

        # Obstáculo = o que a segmentação viu e o YOLO não explicou como bola/trave/robô.
        known_boxes = [d.xyxy for d in accepted if d.yolo_class in FIELD_OBJECT_CLASSES]
        obstacles = fuse_obstacles_with_detections(segmentation.obstacles, known_boxes)
        self.pub_obstacles.publish(build_obstacle_array_message(header, obstacles))

    def _publish_debug(self, header: Header, frame, segmentation: PipelineResult,
                       verdicts: Sequence[Verdict]) -> None:
        wants_topic = self.debug_pub.get_subscription_count() > 0
        if not (wants_topic or self.settings.show_window):
            return  # evita desenhar à toa

        debug_image = draw_debug_image(frame, segmentation, verdicts)
        if wants_topic:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
        if self.settings.show_window:
            cv2.imshow(WINDOW_NAME, debug_image)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Parâmetros e utilitários
    # ------------------------------------------------------------------

    def _on_parameters_changed(self, params) -> SetParametersResult:
        overrides = {param.name: param.value for param in params}
        for name, current in self.startup_values.items():
            if overrides.get(name, current) != current:
                return SetParametersResult(
                    successful=False, reason=f'{name} só pode ser definido na inicialização')
        try:
            settings = read_settings(self, overrides)
        except ValueError as error:
            return SetParametersResult(successful=False, reason=str(error))

        self.settings = settings
        self.pipeline.set_color_detector(settings.color_detector)
        self.pipeline.config = settings.pipeline
        self.validator.color_detector = settings.color_detector
        self.validator.config = settings.validation
        return SetParametersResult(successful=True)

    def _check_model_classes(self, strict: bool) -> None:
        """Garante que o modelo usa o esquema de classes que o nó interpreta.

        Um modelo com outro esquema (ex.: o do robô real, com gols direito/esquerdo e centro)
        seria lido com os significados trocados, então por padrão a partida é recusada.
        """
        self.get_logger().info(f'MODELO: {self.model_file} | CLASSES: {self.model.names}')
        problems = class_name_mismatches(self.model.names)
        if not problems:
            return
        message = ('O modelo não segue o esquema de classes deste nó (ball, goalpost, robot, '
                   'L/T/X-Intersection, crossbar): ' + '; '.join(problems))
        if strict:
            raise ValueError(message + '. Use strict_class_check:=false só se souber o que faz.')
        self.get_logger().warn(message)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSimulacao()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
