"""
detection_validator.py

Valida as detecções do YOLOv8 contra a segmentação do pipeline de visão.

Regra: uma detecção só vale se estiver dentro da área segmentada.
  * bola, trave e robô: o ponto de apoio (base da caixa) deve estar na região
    do campo, que é o convex hull da fronteira detectada (`hull_mask`);
  * interseções (L, T, X): o centro deve estar na região do campo e a caixa
    deve conter pixels da cor das marcações (linhas brancas);
  * travessão: não é publicado (o behavior não o consome), então é ignorado.

Se o pipeline não achou campo nenhum na imagem, nada é validado.
Módulo independente de ROS.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detections import (
    INTERSECTION_CLASSES,
    Detection,
    RejectionReason,
    Verdict,
    YoloClass,
)
from .pipeline import ColorDetector, PipelineResult


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    footpoint_margin_px: int = 12     # tolerância (px da imagem) para o ponto de apoio
    min_marking_ratio: float = 0.04   # fração mínima de pixels de marcação na caixa


class FieldRegion:
    """Região do campo em coordenadas da imagem original.

    A máscara do pipeline fica na resolução de processamento (ex.: 540 px de
    largura), então os pontos são convertidos para essa escala em vez de
    redimensionar a máscara a cada frame.
    """

    def __init__(self, mask: Optional[np.ndarray], frame_shape: Tuple[int, int]):
        self._mask = mask
        frame_h, frame_w = frame_shape[:2]
        self._frame_size = (frame_w, frame_h)
        if mask is None:
            self._scale_x = self._scale_y = 1.0
        else:
            self._scale_x = mask.shape[1] / frame_w
            self._scale_y = mask.shape[0] / frame_h

    @classmethod
    def from_segmentation(cls, segmentation: PipelineResult, frame_shape) -> 'FieldRegion':
        if not segmentation.field_detected:
            return cls(None, frame_shape)
        return cls(segmentation.hull_mask, frame_shape)

    @property
    def is_empty(self) -> bool:
        return self._mask is None

    def contains(self, x: float, y: float, margin_px: int = 0) -> bool:
        """True se há campo no ponto (x, y) ou a até `margin_px` dele."""
        if self._mask is None:
            return False
        frame_w, frame_h = self._frame_size
        x = float(np.clip(x, 0, frame_w - 1))
        y = float(np.clip(y, 0, frame_h - 1))
        mask_x, mask_y = x * self._scale_x, y * self._scale_y
        margin_x, margin_y = margin_px * self._scale_x, margin_px * self._scale_y
        x0, x1 = int(mask_x - margin_x), int(np.ceil(mask_x + margin_x)) + 1
        y0, y1 = int(mask_y - margin_y), int(np.ceil(mask_y + margin_y)) + 1
        window = self._mask[max(0, y0):y1, max(0, x0):x1]
        return bool(window.any())

    def as_frame_mask(self) -> np.ndarray:
        """Máscara na resolução da imagem original (para visualização)."""
        frame_w, frame_h = self._frame_size
        if self._mask is None:
            return np.zeros((frame_h, frame_w), dtype=np.uint8)
        return cv2.resize(self._mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)


class DetectionValidator:
    def __init__(self, color_detector: ColorDetector, config: Optional[ValidationConfig] = None):
        self.color_detector = color_detector
        self.config = config or ValidationConfig()

    def validate(
        self,
        detections: Sequence[Detection],
        segmentation: PipelineResult,
        frame_bgr: np.ndarray,
    ) -> List[Verdict]:
        """Devolve um `Verdict` por detecção, na mesma ordem."""
        if not self.config.enabled:
            return [Verdict(d, self._rejection_for_unsupported(d)) for d in detections]

        region = FieldRegion.from_segmentation(segmentation, frame_bgr.shape)
        marking_mask = None  # calculada só se houver interseções (é a parte cara)

        verdicts = []
        for detection in detections:
            if detection.yolo_class in INTERSECTION_CLASSES and marking_mask is None:
                marking_mask = self.color_detector.marking_mask(frame_bgr)
            rejection = self._rejection_for(detection, region, marking_mask)
            verdicts.append(Verdict(detection, rejection))
        return verdicts

    # ------------------------------------------------------------------

    @staticmethod
    def _rejection_for_unsupported(detection: Detection) -> Optional[RejectionReason]:
        """Com a validação desligada, só a classe sem mensagem ROS é descartada."""
        if detection.yolo_class == YoloClass.CROSSBAR:
            return RejectionReason.IGNORED_CLASS
        return None

    def _rejection_for(
        self,
        detection: Detection,
        region: FieldRegion,
        marking_mask: Optional[np.ndarray],
    ) -> Optional[RejectionReason]:
        if detection.yolo_class == YoloClass.CROSSBAR:
            return RejectionReason.IGNORED_CLASS
        if region.is_empty:
            return RejectionReason.NO_FIELD_DETECTED

        margin = self.config.footpoint_margin_px
        if detection.yolo_class in INTERSECTION_CLASSES:
            if not region.contains(detection.center_x, detection.center_y, margin):
                return RejectionReason.OUTSIDE_FIELD
            if self._marking_ratio(detection, marking_mask) < self.config.min_marking_ratio:
                return RejectionReason.NO_MARKING
            return None

        if not region.contains(*detection.footpoint, margin_px=margin):
            return RejectionReason.OUTSIDE_FIELD
        return None

    @staticmethod
    def _marking_ratio(detection: Detection, marking_mask: np.ndarray) -> float:
        """Fração da caixa (recortada à imagem) coberta por pixels de marcação."""
        frame_h, frame_w = marking_mask.shape[:2]
        x0, y0, x1, y1 = detection.xyxy
        x0, x1 = int(max(0, x0)), int(min(frame_w, np.ceil(x1)))
        y0, y1 = int(max(0, y0)), int(min(frame_h, np.ceil(y1)))
        crop = marking_mask[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        return float(np.count_nonzero(crop)) / crop.size
