"""
vision_parameters.py

Declaração e leitura dos parâmetros ROS do nó de visão.

`read_settings` valida tudo e devolve um `VisionSettings` imutável. Ela serve
tanto para o início do nó quanto para o callback de `ros2 param set`: nesse
caso os valores propostos chegam em `overrides` (o `get_parameter` ainda
devolveria os antigos) e, se algum for inválido, a mudança é recusada inteira.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

from .detection_validator import ValidationConfig
from .pipeline import (
    ColorDetector,
    ColorSpaceHSV,
    PipelineConfig,
    build_color_detector,
    color_spaces_from_calibration,
)

if TYPE_CHECKING:  # só para tipagem: mantém este módulo importável sem ROS
    from rclpy.node import Node

# Classes de cor ajustáveis em runtime (as marcações usam o mesmo espaço do branco).
HSV_CLASSES = ('field', 'white', 'red', 'blue')

# Branco: S e V vêm de def_white_threshold.py (média ± 2σ de white_pixels.csv).
# O hue fica livre (0-179): com saturação ~0 ele é só ruído de renderização, e
# a faixa estreita do CSV deixava a detecção das linhas sensível à iluminação.
WHITE_HSV_MIN = (0, 0, 213)
WHITE_HSV_MAX = (179, 43, 255)


@dataclass(frozen=True)
class VisionSettings:
    color_detector: ColorDetector
    pipeline: PipelineConfig
    validation: ValidationConfig
    show_window: bool
    publish_segmentation: bool


def default_hsv_ranges(green_csv_path: str) -> Dict[str, ColorSpaceHSV]:
    """Faixas HSV iniciais: campo vindo do CSV de calibração, branco fixo."""
    return color_spaces_from_calibration(green_csv_path, WHITE_HSV_MIN, WHITE_HSV_MAX)


def declare_parameters(node: 'Node', hsv_defaults: Mapping[str, ColorSpaceHSV]) -> None:
    node.declare_parameter('show_window', True)
    node.declare_parameter('publish_segmentation', True)

    validation = ValidationConfig()
    node.declare_parameter('validation.enabled', validation.enabled)
    node.declare_parameter('validation.footpoint_margin_px', validation.footpoint_margin_px)
    node.declare_parameter('validation.min_marking_ratio', validation.min_marking_ratio)

    pipeline = PipelineConfig()
    node.declare_parameter('pipeline.process_width', pipeline.process_width)
    node.declare_parameter('pipeline.column_step', pipeline.column_step)
    node.declare_parameter('pipeline.min_obstacle_area', pipeline.min_obstacle_area)
    node.declare_parameter('pipeline.head_tilted_up', pipeline.head_tilted_up)

    for name in HSV_CLASSES:
        low, high = hsv_defaults[name].as_min_max()
        node.declare_parameter(f'hsv.{name}_min', low)
        node.declare_parameter(f'hsv.{name}_max', high)


def read_settings(node: 'Node', overrides: Optional[Mapping[str, Any]] = None) -> VisionSettings:
    """Lê os parâmetros do nó; `overrides` tem prioridade. Levanta ValueError se inválidos."""
    overrides = overrides or {}

    def value(name: str) -> Any:
        return overrides[name] if name in overrides else node.get_parameter(name).value

    spaces = {
        name: _hsv_space(name, value(f'hsv.{name}_min'), value(f'hsv.{name}_max'))
        for name in HSV_CLASSES
    }
    spaces['marking'] = spaces['white']

    return VisionSettings(
        color_detector=build_color_detector(spaces),
        pipeline=PipelineConfig(
            process_width=_at_least('pipeline.process_width', value('pipeline.process_width'), 0),
            column_step=_at_least('pipeline.column_step', value('pipeline.column_step'), 1),
            min_obstacle_area=_at_least(
                'pipeline.min_obstacle_area', value('pipeline.min_obstacle_area'), 0),
            n_line_samples=0,  # pontos de linha não são publicados nesta versão
            head_tilted_up=bool(value('pipeline.head_tilted_up')),
        ),
        validation=ValidationConfig(
            enabled=bool(value('validation.enabled')),
            footpoint_margin_px=_at_least(
                'validation.footpoint_margin_px', value('validation.footpoint_margin_px'), 0),
            min_marking_ratio=_ratio(
                'validation.min_marking_ratio', value('validation.min_marking_ratio')),
        ),
        show_window=bool(value('show_window')),
        publish_segmentation=bool(value('publish_segmentation')),
    )


# ----------------------------------------------------------------------
# Validação de valores
# ----------------------------------------------------------------------

def _hsv_space(name: str, low: Sequence[int], high: Sequence[int]) -> ColorSpaceHSV:
    low, high = list(low), list(high)
    for label, triple in ((f'hsv.{name}_min', low), (f'hsv.{name}_max', high)):
        if len(triple) != 3:
            raise ValueError(f'{label} deve ter 3 valores [H, S, V]')
        if not 0 <= triple[0] <= 179:
            raise ValueError(f'{label}: H deve estar entre 0 e 179')
        if not (0 <= triple[1] <= 255 and 0 <= triple[2] <= 255):
            raise ValueError(f'{label}: S e V devem estar entre 0 e 255')
    return ColorSpaceHSV.from_min_max(low, high)


def _at_least(name: str, number: int, minimum: int) -> int:
    if number < minimum:
        raise ValueError(f'{name} deve ser >= {minimum}')
    return int(number)


def _ratio(name: str, number: float) -> float:
    if not 0.0 <= number <= 1.0:
        raise ValueError(f'{name} deve estar entre 0 e 1')
    return float(number)
