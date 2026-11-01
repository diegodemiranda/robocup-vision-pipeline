#!/usr/bin/env python3
# coding=utf-8
"""
running_inference.py

Carrega o modelo YOLO e roda a inferência em um frame.

Funciona com qualquer modelo detector do Ultralytics que tenha as 7 classes do
projeto: YOLOv8n (`best.pt`, versão 1) ou YOLO26n (versão 2). O YOLO26 é
NMS-free (cabeça one-to-one), mas devolve o mesmo objeto `Results`, então o
restante da visão não muda.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

IMAGE_SIZE = 320
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5   # só vale para modelos com NMS (YOLOv8); o YOLO26 ignora
MAX_DETECTIONS = 10
DEFAULT_MODEL_FILE = 'best.pt'


def load_model(model_file: str = DEFAULT_MODEL_FILE, model_dir: Optional[str] = None) -> YOLO:
    """Carrega um modelo `.pt`.

    `model_file` é o nome de um arquivo em `share/Webots_YOLOv8/modelo/` ou um
    caminho absoluto. Levanta FileNotFoundError listando os modelos disponíveis,
    para o nó falhar logo na partida em vez de na primeira imagem.
    """
    if model_dir is None:
        from ament_index_python.packages import get_package_share_directory
        model_dir = os.path.join(get_package_share_directory('Webots_YOLOv8'), 'modelo')

    model_path = model_file if os.path.isabs(model_file) else os.path.join(model_dir, model_file)
    if not os.path.isfile(model_path):
        available = sorted(f for f in os.listdir(model_dir) if f.endswith('.pt')) \
            if os.path.isdir(model_dir) else []
        raise FileNotFoundError(
            f'Modelo não encontrado: {model_path}. Disponíveis em {model_dir}: {available}')

    print(f'Carregando o modelo de: {model_path}')
    return YOLO(model_path)


def detect_model(
    model: YOLO,
    current_frame: np.ndarray,
    conf: float = CONFIDENCE_THRESHOLD,
    imgsz: int = IMAGE_SIZE,
    draw: bool = True,
) -> Tuple[List[float], List[float], List[List[float]], np.ndarray]:
    """Realiza a inferência em um frame BGR.

    Devolve (classes, scores, boxes_xywh, frame_com_deteccoes), com as caixas em
    pixels da imagem original no formato [centro_x, centro_y, largura, altura].
    Com `draw=False` o último item é o próprio frame, sem o custo de desenhar.
    """
    results = model.predict(
        source=current_frame,
        conf=conf,
        imgsz=imgsz,
        max_det=MAX_DETECTIONS,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    boxes = results[0].boxes
    annotated = results[0].plot() if draw else current_frame
    return boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywh.tolist(), annotated
