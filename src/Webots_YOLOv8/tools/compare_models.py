#!/usr/bin/env python3
"""
compare_models.py

Compara modelos YOLO do projeto (ex.: YOLOv8n `best.pt` x YOLO26n) nas mesmas condições
do nó de visão: imgsz, conf e max_det de `running_inference.py`.

Mede, para cada modelo:
  * parâmetros, GFLOPs (modelo fundido) e tamanho do arquivo;
  * latência por frame (pré-processamento, inferência, pós-processamento e total),
    alternando os modelos em rodadas para diluir ruído da máquina;
  * (--onnx) a mesma latência depois de exportar para ONNX;
  * (--data) acurácia no conjunto de validação/teste: precisão, recall, mAP50, mAP50-95;
  * (--pipeline) o custo do restante da visão por frame (segmentação, validação, debug).

Exemplos:
  # velocidade, a partir de pesos treinados
  python tools/compare_models.py --models modelo/best.pt modelo/yolo26n_best.pt

  # arquitetura sem pesos treinados (7 classes), só para latência
  python tools/compare_models.py --models yolov8n.yaml yolo26n.yaml --nc 7

  # acurácia, no imgsz usado em produção (320)
  python tools/compare_models.py --models modelo/best.pt modelo/yolo26n_best.pt \
      --data /caminho/bitbotstorso_dataset/data.yaml --split test
"""

import argparse
import copy
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import Webots_YOLOv8.running_inference as ri  # noqa: E402
from Webots_YOLOv8.detections import EXPECTED_CLASS_NAMES  # noqa: E402

CAMERA_SIZE = (1080, 720)  # (largura, altura) da câmera do AUREA


@dataclass
class ModelReport:
    label: str
    spec: str
    params_m: float = 0.0
    gflops: float = 0.0
    gflops_640: float = 0.0
    file_mb: Optional[float] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)
    onnx_latency_ms: Dict[str, float] = field(default_factory=dict)
    accuracy: Dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Modelos e frames
# ----------------------------------------------------------------------

def load_model(spec: str, nc: int, out_dir: Path) -> YOLO:
    """`.pt` carrega os pesos; `.yaml` monta só a arquitetura (pesos NÃO treinados).

    A arquitetura é salva em um `.pt` em `out_dir` e reaberta, para ficar idêntica a um
    modelo carregado de checkpoint (necessário para exportar).
    """
    if not spec.endswith('.yaml'):
        return YOLO(spec)
    model = YOLO(spec)
    model.model = DetectionModel(spec, nc=nc, verbose=False)
    model.model.names = (
        {int(k): v for k, v in EXPECTED_CLASS_NAMES.items()} if nc == 7
        else {i: str(i) for i in range(nc)})
    checkpoint = out_dir / f'{Path(spec).stem}_nc{nc}_untrained.pt'
    model.save(checkpoint)
    return YOLO(str(checkpoint))


def synthetic_frames(count: int = 8) -> List[np.ndarray]:
    """Cenas sintéticas na resolução da câmera (céu, campo, linhas e dois objetos)."""
    width, height = CAMERA_SIZE
    frames = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        frame = np.full((height, width, 3), (220, 170, 110), np.uint8)
        horizon = 330 + 10 * seed
        field_color = np.array((60, 140, 50), np.int16)
        noise = rng.integers(-8, 9, (height - horizon, width, 3), dtype=np.int16)
        frame[horizon:] = np.clip(field_color + noise, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (0, 520), (width, 526), (245, 245, 245), -1)
        cv2.rectangle(frame, (300 + 20 * seed, horizon), (306 + 20 * seed, height), (245, 245, 245), -1)
        cv2.circle(frame, (500 + 30 * seed, 600), 18, (240, 240, 240), -1)
        cv2.rectangle(frame, (850, 300), (920, 470), (40, 40, 200), -1)
        frames.append(frame)
    return frames


def load_frames(images_dir: Optional[str]) -> List[np.ndarray]:
    if not images_dir:
        return synthetic_frames()
    paths = sorted(p for p in Path(images_dir).iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    frames = [cv2.imread(str(p)) for p in paths]
    return [f for f in frames if f is not None] or synthetic_frames()


# ----------------------------------------------------------------------
# Medições
# ----------------------------------------------------------------------

def model_stats(model: YOLO, spec: str, imgsz: int) -> Dict[str, Optional[float]]:
    """Parâmetros e GFLOPs do modelo FUNDIDO (BatchNorm incorporada), que é o que roda em inferência.

    É também a base dos números publicados pelo Ultralytics, o que permite compará-los.
    """
    fused = copy.deepcopy(model.model).eval().fuse(verbose=False)
    path = Path(getattr(model, 'ckpt_path', None) or spec)
    return {
        'params_m': sum(p.numel() for p in fused.parameters()) / 1e6,
        'gflops': float(get_flops(fused, imgsz)),
        'gflops_640': float(get_flops(fused, 640)),
        'file_mb': path.stat().st_size / 1e6 if path.is_file() else None,
    }


def summarize(samples: List[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    return {
        'mean': statistics.fmean(samples),
        'std': statistics.pstdev(samples),
        'p50': ordered[len(ordered) // 2],
        'p95': ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def benchmark_latency(models: Dict[str, YOLO], frames: List[np.ndarray], imgsz: int,
                      runs: int, rounds: int, warmup: int) -> Dict[str, Dict[str, float]]:
    """Alterna os modelos a cada rodada; devolve estatísticas por fase em milissegundos."""
    for model in models.values():
        for index in range(warmup):
            ri.detect_model(model, frames[index % len(frames)], imgsz=imgsz, draw=False)

    samples = {label: {'preprocess': [], 'inference': [], 'postprocess': [], 'total': []}
               for label in models}
    per_round = max(1, runs // rounds)
    for _ in range(rounds):
        for label, model in models.items():
            for index in range(per_round):
                frame = frames[index % len(frames)]
                start = time.perf_counter()
                results = model.predict(
                    source=frame, conf=ri.CONFIDENCE_THRESHOLD, imgsz=imgsz,
                    max_det=ri.MAX_DETECTIONS, iou=ri.IOU_THRESHOLD, verbose=False)
                _ = results[0].boxes.xywh.tolist()
                total = (time.perf_counter() - start) * 1000.0
                for phase in ('preprocess', 'inference', 'postprocess'):
                    samples[label][phase].append(results[0].speed[phase])
                samples[label]['total'].append(total)

    report = {}
    for label, phases in samples.items():
        report[label] = {f'{phase}_{stat}': value
                         for phase, values in phases.items()
                         for stat, value in summarize(values).items()}
        report[label]['runs'] = per_round * rounds
    return report


def export_onnx(model: YOLO, imgsz: int, out_dir: Path) -> YOLO:
    """Exporta para ONNX (CPU) e reabre pelo Ultralytics para medir igual ao .pt.

    O Ultralytics grava o .onnx ao lado dos pesos; ele é movido para `out_dir` para não
    poluir a pasta `modelo/` do pacote.
    """
    exported = Path(model.export(format='onnx', imgsz=imgsz, simplify=False, verbose=False))
    target = out_dir / exported.name
    shutil.move(str(exported), target)
    return YOLO(str(target), task='detect')


def evaluate_accuracy(model: YOLO, data: str, split: str, imgsz: int, device: str) -> Dict[str, float]:
    metrics = model.val(data=data, split=split, imgsz=imgsz, device=device, verbose=False, plots=False)
    box = metrics.box
    return {
        'precision': float(box.mp), 'recall': float(box.mr),
        'map50': float(box.map50), 'map50_95': float(box.map),
        **{f'ap50_95_{name}': float(value)
           for name, value in zip(metrics.names.values(), box.maps)},
    }


def benchmark_pipeline(frames: List[np.ndarray], runs: int) -> Optional[Dict[str, float]]:
    """Custo, por frame, da segmentação, validação e imagem de debug (sem ROS)."""
    from Webots_YOLOv8.detection_validator import DetectionValidator
    from Webots_YOLOv8.detections import Detection, YoloClass
    from Webots_YOLOv8.pipeline import PipelineConfig, VisionPipeline, build_color_detector
    from Webots_YOLOv8.vision_debug import draw_debug_image
    from Webots_YOLOv8.vision_parameters import default_hsv_ranges

    detector = build_color_detector(default_hsv_ranges(str(PACKAGE_ROOT / 'recursos' / 'green_pixels.csv')))
    pipeline = VisionPipeline(detector, PipelineConfig(n_line_samples=0), seed=0)
    validator = DetectionValidator(detector)
    detections = [
        Detection(YoloClass.BALL, 0.9, 500.0, 585.0, 30.0, 30.0),
        Detection(YoloClass.ROBOT, 0.9, 885.0, 385.0, 70.0, 170.0),
        Detection(YoloClass.X_INTERSECTION, 0.8, 303.0, 523.0, 30.0, 30.0),
    ]
    times = {'segmentation': [], 'validation': [], 'debug_image': []}
    for index in range(runs):
        frame = frames[index % len(frames)]
        start = time.perf_counter()
        segmentation = pipeline.run(frame)
        middle = time.perf_counter()
        verdicts = validator.validate(detections, segmentation, frame)
        validated = time.perf_counter()
        draw_debug_image(frame, segmentation, verdicts)
        end = time.perf_counter()
        times['segmentation'].append((middle - start) * 1000.0)
        times['validation'].append((validated - middle) * 1000.0)
        times['debug_image'].append((end - validated) * 1000.0)
    return {f'{name}_{stat}': value for name, values in times.items()
            for stat, value in summarize(values).items()}


# ----------------------------------------------------------------------
# Saída
# ----------------------------------------------------------------------

def markdown_table(reports: List[ModelReport]) -> str:
    lines = ['| Modelo | Params (M) | GFLOPs | Arquivo (MB) | Latência total (ms) | p95 (ms) | FPS |',
             '|---|---|---|---|---|---|---|']
    for report in reports:
        latency = report.latency_ms
        file_mb = f'{report.file_mb:.1f}' if report.file_mb is not None else '-'
        lines.append(
            f'| {report.label} | {report.params_m:.2f} | {report.gflops:.2f} | {file_mb} | '
            f'{latency["total_mean"]:.1f} ± {latency["total_std"]:.1f} | {latency["total_p95"]:.1f} | '
            f'{1000.0 / latency["total_mean"]:.1f} |')
    accuracy_rows = [r for r in reports if r.accuracy]
    if accuracy_rows:
        lines += ['', '| Modelo | Precisão | Recall | mAP50 | mAP50-95 |', '|---|---|---|---|---|']
        for report in accuracy_rows:
            a = report.accuracy
            lines.append(f'| {report.label} | {a["precision"]:.3f} | {a["recall"]:.3f} | '
                         f'{a["map50"]:.3f} | {a["map50_95"]:.3f} |')
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--models', nargs='+', required=True, help='.pt ou .yaml (arquitetura)')
    parser.add_argument('--labels', nargs='+', help='nomes para a tabela (mesma ordem de --models)')
    parser.add_argument('--nc', type=int, default=7, help='classes, ao usar .yaml')
    parser.add_argument('--imgsz', type=int, default=ri.IMAGE_SIZE)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, help='threads do PyTorch (padrão: todos)')
    parser.add_argument('--images', help='pasta com imagens reais (padrão: cenas sintéticas)')
    parser.add_argument('--data', help='data.yaml do dataset, para medir acurácia')
    parser.add_argument('--split', default='val', help='val ou test')
    parser.add_argument('--onnx', action='store_true', help='mede também o modelo exportado em ONNX')
    parser.add_argument('--pipeline', action='store_true', help='mede segmentação, validação e debug')
    parser.add_argument('--output-dir', default='comparison')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    labels = args.labels or [Path(spec).stem for spec in args.models]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {label: load_model(spec, args.nc, out_dir) for label, spec in zip(labels, args.models)}
    frames = load_frames(args.images)
    reports = []
    for label, spec in zip(labels, args.models):
        reports.append(ModelReport(label=label, spec=spec, **{
            k: v for k, v in model_stats(models[label], spec, args.imgsz).items()}))

    print(f'Latência: {args.runs} frames/modelo, {args.rounds} rodadas, imgsz={args.imgsz}, '
          f'threads={torch.get_num_threads()}')
    latency = benchmark_latency(models, frames, args.imgsz, args.runs, args.rounds, args.warmup)
    for report in reports:
        report.latency_ms = latency[report.label]

    if args.onnx:
        onnx_models = {label: export_onnx(models[label], args.imgsz, out_dir) for label in labels}
        onnx_latency = benchmark_latency(onnx_models, frames, args.imgsz, args.runs, args.rounds, args.warmup)
        for report in reports:
            report.onnx_latency_ms = onnx_latency[report.label]

    if args.data:
        for report in reports:
            report.accuracy = evaluate_accuracy(
                models[report.label], args.data, args.split, args.imgsz, args.device)

    result = {'reports': [asdict(r) for r in reports], 'imgsz': args.imgsz,
              'threads': torch.get_num_threads(), 'torch': torch.__version__}
    if args.pipeline:
        result['pipeline_ms'] = benchmark_pipeline(frames, args.runs)

    (out_dir / 'results.json').write_text(json.dumps(result, indent=2))
    table = markdown_table(reports)
    (out_dir / 'results.md').write_text(table + '\n')
    print(table)


if __name__ == '__main__':
    main()
