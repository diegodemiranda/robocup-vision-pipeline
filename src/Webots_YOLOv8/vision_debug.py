"""
vision_debug.py

Imagem de debug: campo segmentado, fronteira, obstáculos e o veredito da
validação de cada detecção do YOLO. Independente de ROS.

Legenda: verde = detecção válida; vermelho = rejeitada (com o motivo);
cinza = classe ignorada.
"""

from typing import Sequence

import cv2
import numpy as np

from .detection_validator import FieldRegion
from .detections import RejectionReason, Verdict
from pipeline import PipelineResult, draw_pipeline_overlay

ACCEPTED_COLOR = (0, 200, 0)       # BGR
REJECTED_COLOR = (0, 0, 255)
IGNORED_COLOR = (160, 160, 160)
FIELD_TINT_COLOR = (255, 160, 0)   # azul claro: área em que as detecções valem
FIELD_TINT_ALPHA = 0.20


def draw_debug_image(
    frame_bgr: np.ndarray,
    segmentation: PipelineResult,
    verdicts: Sequence[Verdict],
) -> np.ndarray:
    """Devolve uma cópia de `frame_bgr` com segmentação e vereditos desenhados."""
    image = _tint_field_region(frame_bgr, segmentation)
    image = draw_pipeline_overlay(image, segmentation)
    for verdict in verdicts:
        _draw_verdict(image, verdict)

    accepted = sum(1 for verdict in verdicts if verdict.accepted)
    summary = f'YOLO: {len(verdicts)} | validas: {accepted}'
    _put_label(image, summary, (20, 30), (255, 255, 255), 0.7)
    return image


def _put_label(image: np.ndarray, text: str, origin, color, scale: float = 0.5) -> None:
    """Texto com halo escuro, legível sobre o campo verde e sobre o céu.

    O halo são cópias do texto deslocadas em 1 px, com a mesma espessura do texto
    (espessuras diferentes alteram o espaçamento dos glifos em algumas versões do OpenCV).
    """
    x, y = origin
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(image, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), 1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def _tint_field_region(frame_bgr: np.ndarray, segmentation: PipelineResult) -> np.ndarray:
    image = frame_bgr.copy()
    region = FieldRegion.from_segmentation(segmentation, frame_bgr.shape)
    if region.is_empty:
        return image
    tinted = image.copy()
    tinted[region.as_frame_mask() > 0] = FIELD_TINT_COLOR
    return cv2.addWeighted(tinted, FIELD_TINT_ALPHA, image, 1.0 - FIELD_TINT_ALPHA, 0)


def _draw_verdict(image: np.ndarray, verdict: Verdict) -> None:
    detection = verdict.detection
    class_name = detection.yolo_class.name.lower()
    if verdict.accepted:
        color = ACCEPTED_COLOR
        label = f'{class_name} {detection.confidence:.2f}'
    else:
        ignored = verdict.rejection == RejectionReason.IGNORED_CLASS
        color = IGNORED_COLOR if ignored else REJECTED_COLOR
        label = f'X {class_name} ({verdict.rejection.value})'

    x0, y0, x1, y1 = (int(round(v)) for v in detection.xyxy)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    _put_label(image, label, (x0, max(12, y0 - 6)), color)
    foot_x, foot_y = detection.footpoint
    cv2.circle(image, (int(round(foot_x)), int(round(foot_y))), 4, color, -1)
