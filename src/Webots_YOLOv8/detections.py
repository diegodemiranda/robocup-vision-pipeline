"""
detections.py

Tipos de dados das detecções do YOLOv8, independentes de ROS.

`running_inference.detect_model` devolve listas paralelas (classes, scores,
boxes). Aqui elas viram objetos `Detection`, que carregam a geometria que o
restante da visão precisa (ponto de apoio, caixa em xyxy) e o veredito da
validação (`Verdict`).
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

BoxXYXY = Tuple[float, float, float, float]


class YoloClass(IntEnum):
    """Classes do modelo `best.pt` (mesma ordem de `model.names`)."""
    BALL = 0
    GOALPOST = 1
    ROBOT = 2
    L_INTERSECTION = 3
    T_INTERSECTION = 4
    X_INTERSECTION = 5
    CROSSBAR = 6


EXPECTED_CLASS_NAMES: Dict[int, str] = {
    YoloClass.BALL: 'ball',
    YoloClass.GOALPOST: 'goalpost',
    YoloClass.ROBOT: 'robot',
    YoloClass.L_INTERSECTION: 'L-Intersection',
    YoloClass.T_INTERSECTION: 'T-Intersection',
    YoloClass.X_INTERSECTION: 'X-Intersection',
    YoloClass.CROSSBAR: 'crossbar',
}

INTERSECTION_CLASSES = frozenset({
    YoloClass.L_INTERSECTION,
    YoloClass.T_INTERSECTION,
    YoloClass.X_INTERSECTION,
})

# Objetos que ficam em pé no campo; "explicam" um obstáculo da segmentação.
FIELD_OBJECT_CLASSES = frozenset({
    YoloClass.BALL,
    YoloClass.GOALPOST,
    YoloClass.ROBOT,
})


@dataclass(frozen=True)
class Detection:
    """Uma caixa do YOLO em pixels da imagem original (centro + tamanho)."""
    yolo_class: YoloClass
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def footpoint(self) -> Tuple[float, float]:
        """Centro da base da caixa; é onde o objeto toca o chão.

        Mesmo critério de `soccer_ipm.utils.bb_footpoint`, que é o ponto
        projetado no plano do campo pelo IPM.
        """
        return self.center_x, self.center_y + self.height / 2.0

    @property
    def xyxy(self) -> BoxXYXY:
        half_w, half_h = self.width / 2.0, self.height / 2.0
        return (
            self.center_x - half_w,
            self.center_y - half_h,
            self.center_x + half_w,
            self.center_y + half_h,
        )


class RejectionReason(Enum):
    """Por que uma detecção do YOLO não foi validada (texto ASCII, vai pro debug)."""
    NO_FIELD_DETECTED = 'no_field'
    OUTSIDE_FIELD = 'outside_field'
    NO_MARKING = 'no_marking'
    IGNORED_CLASS = 'ignored_class'


@dataclass(frozen=True)
class Verdict:
    """Resultado da validação de uma detecção."""
    detection: Detection
    rejection: Optional[RejectionReason] = None

    @property
    def accepted(self) -> bool:
        return self.rejection is None


def detections_from_yolo(
    classes: Sequence[float],
    scores: Sequence[float],
    boxes_xywh: Sequence[Sequence[float]],
) -> List[Detection]:
    """Converte a saída de `running_inference.detect_model` em `Detection`.

    Ids de classe que não existem em `YoloClass` são descartados.
    """
    detections: List[Detection] = []
    for class_id, score, (cx, cy, w, h) in zip(classes, scores, boxes_xywh):
        try:
            yolo_class = YoloClass(int(class_id))
        except ValueError:
            continue
        detections.append(Detection(yolo_class, float(score), cx, cy, w, h))
    return detections


def class_name_mismatches(model_names: Dict[int, str]) -> List[str]:
    """Lista divergências entre `model.names` e o mapa esperado pelo código."""
    problems = []
    for class_id, expected in EXPECTED_CLASS_NAMES.items():
        actual = model_names.get(int(class_id))
        if actual != expected:
            problems.append(f'classe {int(class_id)}: esperado {expected!r}, modelo tem {actual!r}')
    return problems
