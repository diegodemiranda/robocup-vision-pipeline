# vision_robot.launch.py: nó de visão + soccer_ipm para o ROBÔ REAL (sem simulador).
#
#   ros2 launch Webots_YOLOv8 vision_robot.launch.py model_file:=/abs/best.pt \
#       camera_topic:=/camera/image_raw camera_info_topic:=/camera/camera_info

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SIM_CAMERA_TOPIC = '/AUREA/camera_optical_frame/image_color'  # fixo em yolo_simulation.py


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('Webots_YOLOv8'), 'config', 'vision_params.yaml')

    args = [
        DeclareLaunchArgument('model_file', default_value='best.pt'),
        DeclareLaunchArgument('show_window', default_value='false'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('camera_frame', default_value='camera_optical_frame'),
        DeclareLaunchArgument('output_frame', default_value='base_link'),
        DeclareLaunchArgument('use_ipm', default_value='true'),
        # TF: só se o robô NÃO publica base_link -> camera_optical_frame (veja o README).
        DeclareLaunchArgument('publish_tf', default_value='false'),
        DeclareLaunchArgument('cam_x', default_value='0.0'),      # m, em base_link (x à frente)
        DeclareLaunchArgument('cam_y', default_value='0.0'),      # m (y à esquerda)
        DeclareLaunchArgument('cam_z', default_value='0.5'),      # m (z para cima)
        DeclareLaunchArgument('cam_roll', default_value='0.0'),   # rad
        DeclareLaunchArgument('cam_pitch', default_value='0.0'),  # rad (positivo = olha para baixo)
        DeclareLaunchArgument('cam_yaw', default_value='0.0'),    # rad
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('camera_id', default_value='0'),       # /dev/video<id>
        DeclareLaunchArgument('width', default_value='1280'),
        DeclareLaunchArgument('height', default_value='720'),
        DeclareLaunchArgument('fps', default_value='30.0'),
        DeclareLaunchArgument('hfov_deg', default_value='90.0'),     # estimativa; calibre a câmera
    ]

    # base_link -> camera_link (pose da câmera no corpo) -> camera_optical_frame (convenção óptica:
    # z à frente, x à direita, y para baixo; rotação fixa roll=-90°, yaw=-90°).
    tf_body = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_tf_body',
        condition=IfCondition(LC('publish_tf')),
        arguments=['--x', LC('cam_x'), '--y', LC('cam_y'), '--z', LC('cam_z'),
                   '--roll', LC('cam_roll'), '--pitch', LC('cam_pitch'), '--yaw', LC('cam_yaw'),
                   '--frame-id', LC('output_frame'), '--child-frame-id', 'camera_link'],
    )
    tf_optical = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_tf_optical',
        condition=IfCondition(LC('publish_tf')),
        arguments=['--roll', '-1.5707963', '--pitch', '0', '--yaw', '-1.5707963',
                   '--frame-id', 'camera_link', '--child-frame-id', LC('camera_frame')],
    )

    camera = Node(
        package='Webots_YOLOv8',
        executable='usb_camera',
        name='usb_camera',
        output='screen',
        condition=IfCondition(LC('start_camera')),
        parameters=[{
            'camera_id': ParameterValue(LC('camera_id'), value_type=int),
            'width': ParameterValue(LC('width'), value_type=int),
            'height': ParameterValue(LC('height'), value_type=int),
            'fps': ParameterValue(LC('fps'), value_type=float),
            'hfov_deg': ParameterValue(LC('hfov_deg'), value_type=float),
            'frame_id': ParameterValue(LC('camera_frame'), value_type=str),
            'image_topic': ParameterValue(LC('camera_topic'), value_type=str),
            'info_topic': ParameterValue(LC('camera_info_topic'), value_type=str),
        }],
    )

    vision = Node(
        package='Webots_YOLOv8',
        executable='finder',
        name='vision_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            config_file,
            {
                'use_sim_time': False,
                'show_window': ParameterValue(LC('show_window'), value_type=bool),
                'model_file': ParameterValue(LC('model_file'), value_type=str),
            },
        ],
        remappings=[(SIM_CAMERA_TOPIC, LC('camera_topic'))],
    )

    ipm = Node(
        package='soccer_ipm',
        executable='ipm',
        name='soccer_ipm_node',
        output='screen',
        condition=IfCondition(LC('use_ipm')),
        parameters=[
            {'use_sim_time': False},
            {'output_frame': ParameterValue(LC('output_frame'), value_type=str)},
            {'balls.ball_diameter': 0.14},
            {'use_distortion': False},
        ],
        remappings=[
            ('camera_info', LC('camera_info_topic')),
        ],
    )

    return LaunchDescription(args + [tf_body, tf_optical, camera, vision, ipm])
