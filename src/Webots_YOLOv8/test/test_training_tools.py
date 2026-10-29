"""A receita de treino do YOLO26n deve ser a do best.pt (YOLOv8n)."""

import os
import sys

import pytest

pytest.importorskip('torch')
pytest.importorskip('ultralytics')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from train_yolo26n import RECIPE_KEYS, recipe_from_checkpoint  # noqa: E402

BEST_PT = os.path.join(os.path.dirname(__file__), '..', 'modelo', 'best.pt')


def test_recipe_matches_how_best_pt_was_trained():
    recipe = recipe_from_checkpoint(BEST_PT)
    assert recipe['epochs'] == 150
    assert recipe['imgsz'] == 640
    assert recipe['batch'] == 16
    assert recipe['seed'] == 0
    assert recipe['patience'] == 50
    assert set(recipe) <= set(RECIPE_KEYS)


def test_recipe_does_not_pin_the_model_or_dataset():
    recipe = recipe_from_checkpoint(BEST_PT)
    assert not {'model', 'data', 'project', 'name', 'device'} & set(recipe)
