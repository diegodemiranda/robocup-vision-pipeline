import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'Webots_YOLOv8'

setup(
    name=package_name,
    version='0.1.0',

    # Layout `src/`: `pipeline` (segmentação, sem ROS) e `Webots_YOLOv8` (integração ROS 2) são
    # pacotes irmãos. Um único diretório-raiz de pacotes é o que mantém o
    # `colcon build --symlink-install` funcionando também com o setuptools antigo do Humble.
    packages=find_packages('src'),
    package_dir={'': 'src'},

    # data_files informa ao colcon quais arquivos adicionais instalar em share/<pacote>/.
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'recursos'), glob('recursos/*.csv')),
        # Pesos NÃO acompanham o repositório (veja o README). Se você colocar .pt/.onnx em
        # modelo/ ANTES do `colcon build`, eles são instalados e `model_file:=nome.pt` os acha.
        (os.path.join('share', package_name, 'modelo'), glob('modelo/*.pt') + glob('modelo/*.onnx')),
    ],

    # Segmentação (pacote `pipeline`): numpy, OpenCV e pandas. O detector é opcional.
    install_requires=['setuptools', 'numpy', 'opencv-python', 'pandas'],
    extras_require={
        'detector': ['ultralytics'],
        'onnx': ['onnx', 'onnxruntime'],
        'dev': ['pytest'],
    },
    zip_safe=True,

    maintainer='Diego de Miranda',
    maintainer_email='diegodemiranda@users.noreply.github.com',
    description='Visão para RoboCup: YOLO + segmentação por cor (HSV) com validação, em ROS 2.',
    license='MIT',

    entry_points={
        'console_scripts': [
            'finder = Webots_YOLOv8.yolo_simulation:main',
        ],
    },
)
