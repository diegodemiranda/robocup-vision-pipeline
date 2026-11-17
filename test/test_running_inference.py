"""
Contrato de inferência: YOLOv8n (versão 1) e YOLO26n (versão 2) funcionam igual para o resto
da visão. Usa as arquiteturas reais de 7 classes (sem pesos treinados, sem download).
Requer o `ultralytics`; sem ele, o módulo é pulado.
"""

import os

import numpy as np
import pytest

pytest.importorskip('ultralytics')

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

import Webots_YOLOv8.running_inference as ri  # noqa: E402
from Webots_YOLOv8.detections import (  # noqa: E402
    EXPECTED_CLASS_NAMES,
    class_name_mismatches,
    detections_from_yolo,
)

ARCHITECTURES = ['yolov8n.yaml', 'yolo26n.yaml']
FRAME = (np.random.default_rng(0).random((720, 1080, 3)) * 255).astype(np.uint8)


@pytest.fixture(scope='module', params=ARCHITECTURES)
def model(request):
    try:
        yolo = YOLO(request.param)
    except Exception as error:  # ex.: Ultralytics antigo, sem o YOLO26
        pytest.skip(f'{request.param} indisponível nesta versão do Ultralytics: {error}')
    yolo.model = DetectionModel(request.param, nc=7, verbose=False)
    yolo.model.names = {int(k): v for k, v in EXPECTED_CLASS_NAMES.items()}
    return yolo


def test_detect_model_returns_parallel_lists_in_the_expected_format(model):
    classes, scores, boxes, annotated = ri.detect_model(model, FRAME, conf=0.0)

    assert 0 < len(classes) <= ri.MAX_DETECTIONS
    assert len(classes) == len(scores) == len(boxes)
    assert all(0 <= int(c) <= 6 for c in classes)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(len(box) == 4 and box[2] > 0 and box[3] > 0 for box in boxes)  # xywh
    assert annotated.shape == FRAME.shape


def test_draw_false_skips_the_annotation_and_returns_the_frame(model):
    *_, annotated = ri.detect_model(model, FRAME, conf=0.0, draw=False)
    assert annotated is FRAME


def test_default_threshold_is_never_looser_than_the_requested_one(model):
    strict = ri.detect_model(model, FRAME, conf=0.9, draw=False)[0]
    loose = ri.detect_model(model, FRAME, conf=0.0, draw=False)[0]
    assert len(strict) <= len(loose)


def test_model_output_feeds_the_rest_of_the_vision(model):
    classes, scores, boxes, _ = ri.detect_model(model, FRAME, conf=0.0, draw=False)
    detections = detections_from_yolo(classes, scores, boxes)
    assert len(detections) == len(classes)   # nenhuma classe fora do mapa do projeto
    assert class_name_mismatches(model.names) == []


class _FakeYOLO:
    def __init__(self, path):
        self.path = path


def test_load_model_resolves_names_in_the_model_dir_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, 'YOLO', _FakeYOLO)
    (tmp_path / 'yolo26n_best.pt').write_bytes(b'x')

    assert ri.load_model('yolo26n_best.pt', model_dir=str(tmp_path)).path == str(
        tmp_path / 'yolo26n_best.pt')
    absolute = str(tmp_path / 'yolo26n_best.pt')
    assert ri.load_model(absolute, model_dir='/outro/lugar').path == absolute


def test_load_model_error_lists_the_available_models(tmp_path):
    (tmp_path / 'best.pt').write_bytes(b'x')
    (tmp_path / 'yolo26n_best.pt').write_bytes(b'x')

    with pytest.raises(FileNotFoundError) as error:
        ri.load_model('faltando.pt', model_dir=str(tmp_path))
    message = str(error.value)
    assert 'faltando.pt' in message and 'best.pt' in message and 'yolo26n_best.pt' in message
    assert os.path.basename(str(tmp_path)) in message
