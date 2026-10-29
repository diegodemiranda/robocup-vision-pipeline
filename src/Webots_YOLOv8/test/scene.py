"""Cena sintética com as cores do Webots, usada pelos testes (câmera 1080x720, horizontal)."""

import cv2
import numpy as np

WIDTH, HEIGHT, HORIZON = 1080, 720, 360

# Posições na cena (pixels da imagem original)
LINE_ROW = (520, 526)       # linha branca horizontal
LINE_COLUMN = (300, 306)    # linha branca vertical
CROSSING = (303, 523)       # cruzamento das duas linhas
GOALPOST_BOX = (700, 250, 724, 480)    # base em y=480, sobre o campo
ROBOT_BOX = (850, 300, 920, 470)       # base em y=470, sobre o campo


def _bgr(h, s, v):
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()


def make_scene(with_field: bool = True, seed: int = 0) -> np.ndarray:
    """Céu em cima, campo verde embaixo, duas linhas, uma trave e um robô vermelho."""
    rng = np.random.default_rng(seed)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = _bgr(105, 120, 220)
    if not with_field:
        return image

    field = np.zeros((HEIGHT - HORIZON, WIDTH, 3), np.int16)
    field[:] = _bgr(57, 160, 140)
    field += rng.integers(-6, 7, field.shape, dtype=np.int16)
    image[HORIZON:] = np.clip(field, 0, 255).astype(np.uint8)

    white = _bgr(120, 3, 243)
    cv2.rectangle(image, (0, LINE_ROW[0]), (WIDTH - 1, LINE_ROW[1]), white, -1)
    cv2.rectangle(image, (LINE_COLUMN[0], HORIZON), (LINE_COLUMN[1], HEIGHT - 1), white, -1)
    cv2.rectangle(image, GOALPOST_BOX[:2], GOALPOST_BOX[2:], white, -1)
    cv2.rectangle(image, ROBOT_BOX[:2], ROBOT_BOX[2:], _bgr(0, 200, 200), -1)
    return image


def xyxy_to_xywh(box):
    """Caixa (x0, y0, x1, y1) -> [centro_x, centro_y, largura, altura], formato do YOLO."""
    x0, y0, x1, y1 = box
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]
