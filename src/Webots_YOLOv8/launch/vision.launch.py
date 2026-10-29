# vision.launch.py

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('Webots_YOLOv8'),
        'config',
        'vision_pipeline.yaml',
    )

    # ros2 launch Webots_YOLOv8 vision.launch.py show_window:=false
    show_window = DeclareLaunchArgument(
        'show_window',
        default_value='true',
        description='Abre a janela do OpenCV com a imagem de debug',
    )

    # ros2 launch Webots_YOLOv8 vision.launch.py model_file:=yolo26n_best.pt
    model_file = DeclareLaunchArgument(
        'model_file',
        default_value='best.pt',
        description='Pesos em modelo/ (best.pt = YOLOv8n, yolo26n_best.pt = YOLO26n)',
    )

    sim_node = Node(
        package='Webots_YOLOv8',
        executable='finder',
        name='vision_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            config_file,
            {
                'show_window': ParameterValue(LaunchConfiguration('show_window'), value_type=bool),
                'model_file': ParameterValue(LaunchConfiguration('model_file'), value_type=str),
            },
        ],
    )

    soccer_ipm = Node(
        package='soccer_ipm',
        executable='ipm',
        name='soccer_ipm_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'output_frame': 'base_link'},
            {'camera_frame': 'camera_optical_frame'},
            {'balls.ball_diameter': 0.14},
            {'use_distortion': False},
        ],
        remappings=[
            ('camera/image_raw', '/AUREA/camera_optical_frame/image_color'),
            ('camera_info', '/AUREA/camera_optical_frame/camera_info'),
        ],
    )

    return LaunchDescription([
        show_window,
        model_file,
        sim_node,
        soccer_ipm,
    ])
