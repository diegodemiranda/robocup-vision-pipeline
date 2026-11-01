"""A segmentação (pacote `pipeline`) tem de funcionar sem ROS, sem Ultralytics e sem o resto do projeto."""

import os
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')

SCRIPT = textwrap.dedent('''
    import sys

    BLOCKED = ('rclpy', 'cv_bridge', 'ultralytics', 'torch', 'Webots_YOLOv8')

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in BLOCKED:
                raise ImportError('bloqueado no teste de independência: ' + name)

    sys.meta_path.insert(0, Blocker())

    import numpy as np
    from pipeline import DEFAULT_COLOR_SPACES, VisionPipeline, build_color_detector

    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:60] = (220, 170, 110)
    frame[60:] = (60, 140, 50)
    result = VisionPipeline(build_color_detector(DEFAULT_COLOR_SPACES), seed=0).run(frame)

    loaded = sorted(m for m in sys.modules if m.split('.')[0] in BLOCKED)
    assert not loaded, loaded
    assert result.field_detected
    print('ok')
''')


def test_pipeline_runs_without_ros_detector_or_the_rest_of_the_project():
    # `Webots_YOLOv8` está ao lado de `pipeline` em src/, então a independência é garantida pelo
    # bloqueio de imports do SCRIPT (um import indevido levanta ImportError).
    env = {**os.environ, 'PYTHONPATH': SRC}
    done = subprocess.run([sys.executable, '-c', SCRIPT], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'ok'
