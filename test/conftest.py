import os
import sys

import pytest

# Os testes rodam direto da árvore de fontes, sem instalar: `pipeline` e `Webots_YOLOv8` ficam em src/.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline import (  # noqa: E402
    PipelineConfig,
    VisionPipeline,
    build_color_detector,
)
from Webots_YOLOv8.vision_parameters import default_hsv_ranges  # noqa: E402

GREEN_CSV = os.path.join(ROOT, 'recursos', 'green_pixels.csv')


@pytest.fixture(scope='session')
def color_detector():
    return build_color_detector(default_hsv_ranges(GREEN_CSV))


@pytest.fixture
def segment(color_detector):
    """Roda a segmentação em uma imagem com a configuração padrão do nó."""
    def _segment(frame):
        config = PipelineConfig(n_line_samples=0)
        return VisionPipeline(color_detector, config, seed=0).run(frame)
    return _segment


@pytest.fixture
def make_harness(monkeypatch, tmp_path):
    """Fábrica de nós de visão com ROS real; pula o teste se o ROS não estiver carregado."""
    pytest.importorskip('rclpy')
    pytest.importorskip('soccer_vision_2d_msgs.msg')
    import ros_harness
    yield from ros_harness.harness_factory(monkeypatch, tmp_path)
