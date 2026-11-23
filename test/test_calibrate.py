"""Ferramenta de calibração (`python -m pipeline.calibrate`): cálculo das faixas, saídas e comandos sem tela."""

import sys

import cv2
import numpy as np
import pytest

from pipeline import DEFAULT_COLOR_SPACES, VisionPipeline, build_color_detector, load_color_spaces_from_csv
from pipeline import calibrate as cal
from scene import make_scene
from Webots_YOLOv8.vision_parameters import _hsv_space


def hsv_to_bgr_image(h, s, v, shape, rng, noise=(3, 15, 15)):
    """Imagem BGR com pixels HSV ao redor de (h, s, v)."""
    hsv = np.stack([
        np.clip(rng.normal(h, noise[0], shape), 0, 179),
        np.clip(rng.normal(s, noise[1], shape), 0, 255),
        np.clip(rng.normal(v, noise[2], shape), 0, 255)], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dry_grass_frame(seed=0):
    """Campo 'real' amarelado (H=28), fora das faixas de exemplo (H 35-85), com céu em cima."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:200] = hsv_to_bgr_image(105, 120, 220, (200, 640), rng)
    frame[200:] = hsv_to_bgr_image(28, 140, 120, (280, 640), rng)
    return frame


# ---------------------------------------------------------------- cálculo das faixas

def test_hue_range_wraps_around_zero_for_red():
    hues = np.concatenate([np.arange(172, 180), np.arange(0, 6)] * 50)
    h_min, h_max = cal.hue_range(hues, 2, 98, margin=3)
    assert h_min > h_max                              # faixa circular
    space = cal.ColorSpaceHSV(h_min, h_max, 0, 255, 0, 255)
    pixels = np.stack([hues, np.full_like(hues, 200), np.full_like(hues, 200)], axis=1)
    assert cal.coverage(space, pixels) > 0.95


def test_hue_range_plain_interval_when_not_wrapping():
    h_min, h_max = cal.hue_range(np.arange(50, 71).repeat(20), 2, 98, margin=4)
    assert 44 <= h_min <= 52 and 68 <= h_max <= 76 and h_min < h_max


def test_space_from_pixels_covers_samples_and_stays_in_bounds():
    rng = np.random.default_rng(1)
    pixels = np.clip(rng.normal([60, 150, 150], [4, 20, 20], (5000, 3)), 0, 255).astype(np.uint8)
    space = cal.space_from_pixels(pixels)
    assert cal.coverage(space, pixels) > 0.95
    assert 0 <= space.h_min <= space.h_max <= 179
    assert 0 <= space.s_min <= space.s_max <= 255 and 0 <= space.v_min <= space.v_max <= 255


def test_margin_widens_the_measured_range():
    pixels = np.stack([np.full(2000, 60), np.linspace(100, 120, 2000), np.linspace(150, 170, 2000)], axis=1)
    wide = cal.space_from_pixels(pixels, margin=(5, 30, 40))
    tight = cal.space_from_pixels(pixels, margin=(0, 0, 0))
    assert wide.s_min <= 72 and wide.s_max >= 148 and wide.v_min <= 112 and wide.v_max >= 208
    assert tight.s_min >= 99 and tight.s_max <= 121


def test_fit_spaces_ignores_hue_for_white_and_keeps_uncalibrated_classes():
    samples = cal.Samples()
    bluish_white = hsv_to_bgr_image(105, 20, 240, (40, 80), np.random.default_rng(4), noise=(1, 3, 3))
    samples.add_box('white', bluish_white, (0, 0, 80, 40))
    spaces, calibrated, _ = cal.fit_spaces(samples, dict(DEFAULT_COLOR_SPACES))
    assert calibrated == ['white'] and (spaces['white'].h_min, spaces['white'].h_max) == (0, 179)
    assert spaces['field'] == DEFAULT_COLOR_SPACES['field'] and spaces['marking'] == spaces['white']


def test_white_ignores_hue():
    rng = np.random.default_rng(2)
    pixels = np.stack([rng.integers(0, 180, 3000), rng.integers(0, 30, 3000), rng.integers(220, 256, 3000)], axis=1)
    space = cal.space_from_pixels(pixels, hue_free=True)
    assert (space.h_min, space.h_max) == (0, 179)


def test_warnings_flag_a_region_that_is_not_the_class():
    sky = np.full((3000, 3), (105, 120, 220), np.uint8)       # céu amostrado como "campo"
    space = cal.space_from_pixels(sky)
    assert any('não ser grama' in w for w in cal.check_samples('field', sky, space))
    few = sky[:100]
    assert any('pixels amostrados' in w for w in cal.check_samples('field', few, space))


def test_parse_sample_accepts_valid_and_rejects_invalid():
    assert cal.parse_sample('field=0.4,0.8,0.2,0.15') == ('field', (0.4, 0.8, 0.2, 0.15))
    for bad in ('field', 'grass=0.1,0.1,0.1,0.1', 'field=0.1,0.1,0.1', 'field=0.9,0.1,0.3,0.1', 'field=0.1,0.1,0,0.1'):
        with pytest.raises(ValueError):
            cal.parse_sample(bad)


# ---------------------------------------------------------------- saídas

def test_csv_round_trips_through_the_pipeline_loader(tmp_path):
    spaces = dict(DEFAULT_COLOR_SPACES)
    spaces['field'] = cal.ColorSpaceHSV(20, 40, 90, 200, 60, 180)
    path = tmp_path / 'hsv.csv'
    path.write_text(cal.spaces_to_csv(spaces), encoding='utf-8')
    loaded = load_color_spaces_from_csv(str(path))
    assert loaded['field'] == spaces['field']
    assert loaded['marking'] == loaded['white'] == spaces['white']
    assert loaded['red'] == spaces['red']                          # hue circular preservado


def test_ros_yaml_only_has_calibrated_classes_and_passes_node_validation():
    spaces = dict(DEFAULT_COLOR_SPACES)
    spaces['field'] = cal.ColorSpaceHSV(20, 40, 90, 200, 60, 180)
    text = cal.spaces_to_ros_yaml(spaces, ['field'])
    assert 'field_min: [20, 90, 60]' in text and 'field_max: [40, 200, 180]' in text
    assert 'red_min' not in text
    yaml = pytest.importorskip('yaml')
    params = yaml.safe_load(text)['vision_node']['ros__parameters']['hsv']
    assert _hsv_space('field', params['field_min'], params['field_max']) == spaces['field']


# ---------------------------------------------------------------- comandos sem tela

def test_hsv_headless_on_an_image_fixes_a_field_the_defaults_miss(tmp_path):
    frame = dry_grass_frame()
    image = tmp_path / 'campo.png'
    cv2.imwrite(str(image), frame)
    field_rows = slice(200, 480)
    default_mask = DEFAULT_COLOR_SPACES['field'].mask(frame)
    assert np.count_nonzero(default_mask[field_rows]) / default_mask[field_rows].size < 0.1   # o padrão não vê o campo

    out = tmp_path / 'calib'
    code = cal.main(['hsv', '--source', str(image), '--sample', 'field=0.2,0.6,0.5,0.3', '--out', str(out)])
    assert code == 0
    for name in ('hsv.csv', 'hsv_params.yaml', 'hsv_preview.png'):
        assert (out / name).exists()

    spaces = load_color_spaces_from_csv(str(out / 'hsv.csv'))
    mask = spaces['field'].mask(frame)
    assert np.count_nonzero(mask[field_rows]) / mask[field_rows].size > 0.95
    assert np.count_nonzero(mask[:200]) == 0                                                   # céu não entra
    assert VisionPipeline(build_color_detector(spaces), seed=0).run(frame).field_detected


def test_hsv_headless_calibrates_white_on_the_synthetic_scene(tmp_path):
    image = tmp_path / 'cena.png'
    cv2.imwrite(str(image), make_scene())
    out = tmp_path / 'calib'
    code = cal.main(['hsv', '--source', str(image), '--out', str(out),
                     '--sample', 'field=0.40,0.80,0.20,0.15', '--sample', 'white=0.10,0.724,0.18,0.004'])
    assert code == 0
    yaml_text = (out / 'hsv_params.yaml').read_text(encoding='utf-8')
    assert 'field_min' in yaml_text and 'white_min' in yaml_text and 'blue_min' not in yaml_text


def test_hsv_headless_from_camera_discards_warmup_and_samples_the_rest(tmp_path, monkeypatch):
    warmup = [np.zeros((480, 640, 3), np.uint8)] * 3               # exposição ainda instável: tudo preto
    frames = warmup + [dry_grass_frame(seed) for seed in range(5)]

    class FakeCap:
        def __init__(self):
            self.reads = 0

        def read(self):
            frame = frames[self.reads % len(frames)]
            self.reads += 1
            return True, frame

        def release(self):
            self.released = True

    cap = FakeCap()
    monkeypatch.setattr(cal, 'open_camera', lambda *a, **k: cap)
    code = cal.main(['hsv', '--source', '0', '--frames', '5', '--skip', '3', '--out', str(tmp_path),
                     '--sample', 'field=0.2,0.6,0.5,0.3'])
    assert code == 0 and cap.reads == 8 and cap.released
    spaces = load_color_spaces_from_csv(str(tmp_path / 'hsv.csv'))
    assert spaces['field'].v_min >= 40                              # o preto do aquecimento ficou de fora
    assert (tmp_path / 'hsv_preview.png').exists()


def test_snapshot_writes_photo_and_grid(tmp_path, monkeypatch):
    class FakeCap:
        def read(self):
            return True, dry_grass_frame()

        def release(self):
            pass

    monkeypatch.setattr(cal, 'open_camera', lambda *a, **k: FakeCap())
    assert cal.main(['snapshot', '--source', '0', '--skip', '1', '--out', str(tmp_path)]) == 0
    assert (tmp_path / 'snapshot.png').exists() and (tmp_path / 'snapshot_grade.png').exists()


def test_errors_return_code_2_with_a_message(tmp_path, capsys, monkeypatch):
    assert cal.main(['hsv', '--source', str(tmp_path / 'nao_existe.png'), '--sample', 'field=0.1,0.1,0.2,0.2']) == 2
    assert 'ERRO' in capsys.readouterr().err
    assert cal.main(['hsv', '--source', '0', '--sample', 'grama=0.1,0.1,0.2,0.2']) == 2
    if sys.platform.startswith('linux'):
        monkeypatch.delenv('DISPLAY', raising=False)
        monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
        image = tmp_path / 'x.png'
        cv2.imwrite(str(image), dry_grass_frame())
        assert cal.main(['hsv', '--source', str(image)]) == 2                  # interativo sem tela
        assert 'sem tela' in capsys.readouterr().err


# ---------------------------------------------------------------- intrínsecos

BOARD = (9, 6)           # cantos internos; tabuleiro de 10x7 quadrados
SQUARE_M = 0.03
TRUE_K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1]])


def chessboard_pattern(square_px=50, margin=50):
    cols, rows = BOARD[0] + 1, BOARD[1] + 1
    pattern = np.full((rows * square_px + 2 * margin, cols * square_px + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * square_px, margin + c * square_px
                pattern[y:y + square_px, x:x + square_px] = 0
    return pattern


def render_view(pattern, rvec, tvec, square_px=50, margin=50, size=(1280, 720)):
    """Foto do tabuleiro com a câmera TRUE_K (sem distorção). O 1º canto interno é a origem do mundo."""
    h, w = pattern.shape
    origin = margin + square_px
    corners_px = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    world = np.float32([[(u - origin) * SQUARE_M / square_px, (v - origin) * SQUARE_M / square_px, 0]
                        for u, v in corners_px])
    projected, _ = cv2.projectPoints(world, rvec, tvec, TRUE_K, None)
    homography = cv2.getPerspectiveTransform(corners_px, projected.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(pattern, homography, size, flags=cv2.INTER_LINEAR, borderValue=128)


def synthetic_views(count=12, seed=3):
    rng = np.random.default_rng(seed)
    pattern, views = chessboard_pattern(), []
    for _ in range(count):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-0.12, 0.0), rng.uniform(-0.08, 0.0), rng.uniform(0.55, 0.8)])
        views.append(render_view(pattern, rvec, tvec))
    return views


def test_intrinsics_recovers_the_known_camera_matrix_from_synthetic_views():
    sets = []
    for view in synthetic_views():
        corners = cal.find_corners(view, BOARD)
        if corners is not None:
            sets.append(corners)
    assert len(sets) >= 8, 'o tabuleiro sintético deveria ser detectado na maioria das vistas'
    result = cal.calibrate_intrinsics(sets, (1280, 720), BOARD, SQUARE_M)
    assert result['rms'] < 1.0
    assert result['fx'] == pytest.approx(800.0, rel=0.05) and result['fy'] == pytest.approx(800.0, rel=0.05)
    assert result['cx'] == pytest.approx(640.0, abs=20) and result['cy'] == pytest.approx(360.0, abs=20)


def test_intrinsics_command_from_image_files_writes_yaml_and_launch_args(tmp_path):
    folder = tmp_path / 'fotos'
    folder.mkdir()
    for index, view in enumerate(synthetic_views()):
        cv2.imwrite(str(folder / f'{index:02d}.png'), view)
    out = tmp_path / 'calib'
    code = cal.main(['intrinsics', '--images', str(folder / '*.png'), '--board', '9x6', '--square', str(SQUARE_M),
                     '--out', str(out)])
    assert code == 0
    args_line = (out / 'camera_args.txt').read_text(encoding='utf-8').strip()
    assert args_line.startswith('width:=1280 height:=720 fx:=') and 'cy:=' in args_line
    assert 'rms_reprojection_px' in (out / 'intrinsics.yaml').read_text(encoding='utf-8')


def test_intrinsics_rejects_too_few_views_and_square_boards():
    with pytest.raises(ValueError):
        cal.calibrate_intrinsics([np.zeros((54, 1, 2), np.float32)] * 3, (1280, 720), BOARD, SQUARE_M)
    with pytest.raises(ValueError):
        cal.parse_board('6x6')
    assert cal.parse_board('9x6') == (9, 6)


# ---------------------------------------------------------------- modo interativo (janela simulada)

def fake_window(monkeypatch, make_steps):
    """Troca as chamadas de janela do OpenCV; `make_steps(handlers)` devolve a tecla de cada `waitKey`."""
    handlers = {}
    queue = iter(make_steps(handlers))
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a, **k: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda name, fn: handlers.update(callback=fn))
    monkeypatch.setattr(cv2, 'imshow', lambda name, image: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda ms: next(queue)())
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)


def drag(handlers, start, end):
    def action():
        callback = handlers['callback']
        callback(cv2.EVENT_LBUTTONDOWN, start[0], start[1], 0, None)
        callback(cv2.EVENT_MOUSEMOVE, end[0], end[1], 0, None)
        callback(cv2.EVENT_LBUTTONUP, end[0], end[1], 0, None)
        return -1
    return action


def test_gui_drag_samples_a_region_and_s_saves(tmp_path, monkeypatch):
    image = tmp_path / 'campo.png'
    cv2.imwrite(str(image), dry_grass_frame())
    fake_window(monkeypatch, lambda h: [lambda: -1, drag(h, (100, 300), (400, 420)), lambda: ord('p'),
                                        lambda: ord('s'), lambda: ord('q')])
    out = tmp_path / 'calib'
    assert cal.main(['hsv', '--source', str(image), '--out', str(out)]) == 0
    spaces = load_color_spaces_from_csv(str(out / 'hsv.csv'))
    mask = spaces['field'].mask(dry_grass_frame())
    assert np.count_nonzero(mask[200:]) / mask[200:].size > 0.95
    assert (out / 'hsv_params.yaml').exists() and (out / 'hsv_preview.png').exists()


def test_gui_quit_without_saving_writes_nothing(tmp_path, monkeypatch, capsys):
    image = tmp_path / 'campo.png'
    cv2.imwrite(str(image), dry_grass_frame())
    fake_window(monkeypatch, lambda h: [lambda: -1, drag(h, (100, 300), (400, 420)), lambda: ord('q')])
    out = tmp_path / 'calib'
    assert cal.main(['hsv', '--source', str(image), '--out', str(out)]) == 0
    assert not out.exists()
    assert 'saiu sem salvar' in capsys.readouterr().out


def test_gui_x_clears_the_class_and_s_then_does_nothing(tmp_path, monkeypatch):
    image = tmp_path / 'campo.png'
    cv2.imwrite(str(image), dry_grass_frame())
    fake_window(monkeypatch, lambda h: [lambda: -1, drag(h, (100, 300), (400, 420)), lambda: ord('x'),
                                        lambda: ord('s'), lambda: ord('q')])
    out = tmp_path / 'calib'
    assert cal.main(['hsv', '--source', str(image), '--out', str(out)]) == 0
    assert not out.exists()                                       # sem classe calibrada, 's' não grava
