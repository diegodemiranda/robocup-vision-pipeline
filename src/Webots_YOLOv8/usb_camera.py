#!/usr/bin/env python3
"""
usb_camera.py

Nó de câmera USB (V4L2, /dev/video<id>) para o robô real: publica `sensor_msgs/Image` (bgr8) e
`sensor_msgs/CameraInfo`. Sem dependências além de OpenCV e cv_bridge.

O CameraInfo é ESTIMADO a partir de `hfov_deg` (pinhole, sem distorção, centro da imagem).
Para precisão no soccer_ipm, calibre a câmera e passe fx, fy, cx, cy.
"""

import math

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROS_Image


class UsbCamera(Node):

    def __init__(self):
        super().__init__('usb_camera')
        p = self.declare_parameter
        p('camera_id', 0)
        p('width', 1280)
        p('height', 720)
        p('fps', 30.0)
        p('fourcc', 'MJPG')
        p('frame_id', 'camera_optical_frame')
        p('hfov_deg', 90.0)
        p('fx', 0.0)
        p('fy', 0.0)
        p('cx', 0.0)
        p('cy', 0.0)
        p('image_topic', '/camera/image_raw')
        p('info_topic', '/camera/camera_info')

        v = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame_id = v('frame_id')
        cam_id = int(v('camera_id'))

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f'Não foi possível abrir /dev/video{cam_id} '
                '(existe? o usuário está no grupo "video"? outra aplicação a usa?)')
        fourcc = str(v('fourcc'))
        if len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(v('width')))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(v('height')))
        self.cap.set(cv2.CAP_PROP_FPS, float(v('fps')))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'/dev/video{cam_id}: {w}x{h} (pedido {v("width")}x{v("height")})')

        fx = float(v('fx')) or (w / 2.0) / math.tan(math.radians(float(v('hfov_deg'))) / 2.0)
        fy = float(v('fy')) or fx
        cx = float(v('cx')) or w / 2.0
        cy = float(v('cy')) or h / 2.0
        self.info = CameraInfo()
        self.info.width, self.info.height = w, h
        self.info.distortion_model = 'plumb_bob'
        self.info.d = [0.0] * 5
        self.info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.bridge = CvBridge()
        self.pub_img = self.create_publisher(ROS_Image, v('image_topic'), 1)
        self.pub_info = self.create_publisher(CameraInfo, v('info_topic'), 1)
        self.timer = self.create_timer(1.0 / max(float(v('fps')), 1.0), self.tick)

    def tick(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Falha ao ler a câmera', throttle_duration_sec=5.0)
            return
        stamp = self.get_clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        self.info.header = msg.header
        self.pub_info.publish(self.info)
        self.pub_img.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UsbCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
