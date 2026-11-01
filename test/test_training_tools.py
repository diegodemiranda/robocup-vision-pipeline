"""A receita de treino do YOLO26n é lida do checkpoint de referência (sem usar pesos reais)."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('ultralytics')

from Webots_YOLOv8.tools.train_yolo26n import RECIPE_KEYS, recipe_from_checkpoint  # noqa: E402

REFERENCE_TRAIN_ARGS = {   # o que o Ultralytics grava em `train_args` de um best.pt
    'model': 'yolov8n.pt', 'data': '/algum/lugar/data.yaml', 'project': 'runs', 'name': 'ref',
    'device': '0', 'epochs': 150, 'patience': 50, 'imgsz': 640, 'batch': 16, 'seed': 0,
    'workers': 2, 'single_cls': False, 'cos_lr': None, 'close_mosaic': 10, 'deterministic': True,
}


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / 'referencia.pt'
    torch.save({'train_args': dict(REFERENCE_TRAIN_ARGS)}, path)
    return str(path)


def test_recipe_is_read_from_the_reference_checkpoint(checkpoint):
    recipe = recipe_from_checkpoint(checkpoint)
    assert recipe['epochs'] == 150
    assert recipe['imgsz'] == 640
    assert recipe['batch'] == 16
    assert recipe['seed'] == 0
    assert recipe['patience'] == 50
    assert set(recipe) <= set(RECIPE_KEYS)


def test_recipe_skips_unset_values_and_does_not_pin_model_or_dataset(checkpoint):
    recipe = recipe_from_checkpoint(checkpoint)
    assert 'cos_lr' not in recipe                       # valor None não vira argumento
    assert not {'model', 'data', 'project', 'name', 'device'} & set(recipe)
