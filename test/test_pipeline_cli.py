"""`python -m pipeline`: segmentação em imagem e vídeo, sem ROS."""

import cv2
import pytest

from pipeline.__main__ import main
from scene import make_scene

HSV_CSV = """classe,h_min,h_max,s_min,s_max,v_min,v_max
field,{field}
marking,0,179,0,60,180,255
red,170,10,100,255,80,255
blue,100,130,100,255,80,255
white,0,179,0,60,180,255
"""
GOOD_FIELD = '35,85,50,255,30,255'
NO_GREEN_FIELD = '0,5,250,255,250,255'   # faixa que não casa com o verde da cena


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / 'cena.png'
    cv2.imwrite(str(path), make_scene())
    return path


def test_image_is_segmented_and_saved_next_to_the_input(photo, capsys):
    assert main([str(photo), '--seed', '0']) == 0

    saida = photo.with_name('cena_segmentada.png')
    annotated, original = cv2.imread(str(saida)), cv2.imread(str(photo))
    assert annotated.shape == original.shape
    assert (annotated != original).any()                       # o overlay foi desenhado
    assert 'campo detectado em 1 de 1' in capsys.readouterr().out


def test_hsv_csv_changes_the_result(photo, tmp_path, capsys):
    bad = tmp_path / 'sem_verde.csv'
    bad.write_text(HSV_CSV.format(field=NO_GREEN_FIELD))
    assert main([str(photo), '--hsv-csv', str(bad), '-o', str(tmp_path / 'x.png')]) == 0
    assert 'campo detectado em 0 de 1' in capsys.readouterr().out


def test_video_is_processed_frame_by_frame(tmp_path, capsys):
    video = tmp_path / 'jogo.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (1080, 720))
    if not writer.isOpened():
        pytest.skip('este OpenCV não grava AVI/MJPG')
    for seed in range(4):
        writer.write(make_scene(seed=seed))
    writer.release()

    saida = tmp_path / 'jogo_anotado.avi'
    assert main([str(video), '-o', str(saida), '--seed', '0']) == 0
    assert 'Quadros: 4' in capsys.readouterr().out

    capture = cv2.VideoCapture(str(saida))
    count = 0
    while capture.read()[0]:
        count += 1
    assert count == 4


@pytest.mark.parametrize('argv, message', [
    (['nao_existe.png'], 'não consegui abrir a imagem'),
    (['nao_existe.mp4'], 'não consegui abrir o vídeo'),
    (['0'], 'Para câmera use --show'),
])
def test_invalid_sources_fail_with_a_message_not_a_traceback(argv, message, capsys):
    assert main(argv) == 2
    assert message in capsys.readouterr().err
