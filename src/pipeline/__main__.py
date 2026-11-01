r"""
python -m pipeline

Executa a segmentação (fronteira do campo, obstáculos e pontos de linha) em uma imagem, um vídeo
ou uma câmera, SEM ROS e SEM detector de objetos: só precisa de numpy, OpenCV e pandas.

Exemplos:
  python -m pipeline foto.png                        # grava foto_segmentada.png
  python -m pipeline partida.mp4 -o saida.mp4        # vídeo -> vídeo anotado
  python -m pipeline 0 --show                        # câmera 0 ao vivo (q para sair)
  python -m pipeline foto.png --hsv-csv meu_hsv.csv  # faixas HSV calibradas por você

Sem --hsv-csv usa as faixas de EXEMPLO de `color_common.DEFAULT_COLOR_SPACES`: elas precisam ser
calibradas para a sua câmera e iluminação. Formato do CSV (OpenCV: H 0-179, S e V 0-255; o vermelho
pode ter h_min > h_max, pois o matiz é circular):

  classe,h_min,h_max,s_min,s_max,v_min,v_max
  field,35,85,50,255,30,255
  marking,0,179,0,60,180,255
  red,170,10,100,255,80,255
  blue,100,130,100,255,80,255
  white,0,179,0,60,180,255
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .color_common import DEFAULT_COLOR_SPACES, build_color_detector, load_color_spaces_from_csv
from .main_pipeline import PipelineConfig, VisionPipeline, draw_pipeline_overlay

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m pipeline', description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('source', help='imagem, vídeo ou índice de câmera (ex.: 0)')
    parser.add_argument('-o', '--output', type=Path,
                        help='arquivo de saída (padrão: <entrada>_segmentada.<ext>)')
    parser.add_argument('--hsv-csv', type=Path, help='CSV com as faixas HSV (formato no --help)')
    parser.add_argument('--width', type=int, default=PipelineConfig.process_width,
                        help='largura de processamento em px; 0 = sem redução (padrão: %(default)s)')
    parser.add_argument('--column-step', type=int, default=PipelineConfig.column_step,
                        help='1 coluna a cada N na varredura da fronteira (padrão: %(default)s)')
    parser.add_argument('--min-obstacle-area', type=int, default=PipelineConfig.min_obstacle_area,
                        help='área mínima de obstáculo em px² (padrão: %(default)s)')
    parser.add_argument('--line-samples', type=int, default=PipelineConfig.n_line_samples,
                        help='amostras para os pontos de linha; 0 desliga (padrão: %(default)s)')
    parser.add_argument('--head-tilted-up', action='store_true',
                        help='varredura de baixo para cima (câmera inclinada para cima)')
    parser.add_argument('--seed', type=int, help='semente do sorteio dos pontos de linha')
    parser.add_argument('--max-frames', type=int, help='para depois de N quadros')
    parser.add_argument('--show', action='store_true', help='mostra uma janela (q para sair)')
    return parser.parse_args(argv)


def build_pipeline(args: argparse.Namespace) -> VisionPipeline:
    spaces = load_color_spaces_from_csv(str(args.hsv_csv)) if args.hsv_csv else DEFAULT_COLOR_SPACES
    config = PipelineConfig(
        process_width=args.width, column_step=args.column_step,
        min_obstacle_area=args.min_obstacle_area, n_line_samples=args.line_samples,
        head_tilted_up=args.head_tilted_up)
    return VisionPipeline(build_color_detector(spaces), config, seed=args.seed)


def is_image(source: str) -> bool:
    return Path(source).suffix.lower() in IMAGE_SUFFIXES


def frames_from(source: str) -> Iterator[np.ndarray]:
    """Quadros BGR da fonte. Levanta ValueError se a fonte não puder ser aberta."""
    if is_image(source):
        image = cv2.imread(source)
        if image is None:
            raise ValueError(f'não consegui abrir a imagem: {source}')
        yield image
        return

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise ValueError(f'não consegui abrir o vídeo/câmera: {source}')
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def default_output(source: str) -> Optional[Path]:
    if source.isdigit():
        return None                                    # câmera: só janela, a menos que haja -o
    path = Path(source)
    return path.with_name(f'{path.stem}_segmentada{path.suffix if is_image(source) else ".mp4"}')


def open_writer(path: Path, frame: np.ndarray) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*('MJPG' if path.suffix.lower() == '.avi' else 'mp4v'))
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        raise ValueError(f'não consegui criar o vídeo de saída: {path}')
    return writer


def run(args: argparse.Namespace) -> int:
    output = args.output or default_output(args.source)
    if output is None and not args.show:
        print('Para câmera use --show e/ou -o ARQUIVO.', file=sys.stderr)
        return 2

    pipeline = build_pipeline(args)
    writer = None
    frames = with_field = 0
    elapsed = 0.0
    try:
        for frame in frames_from(args.source):
            start = time.perf_counter()
            result = pipeline.run(frame)
            elapsed += time.perf_counter() - start
            annotated = draw_pipeline_overlay(frame, result)
            frames += 1
            with_field += int(result.field_detected)

            if output is not None:
                if is_image(args.source):
                    cv2.imwrite(str(output), annotated)
                else:
                    writer = writer or open_writer(output, annotated)
                    writer.write(annotated)
            if args.show:
                cv2.imshow('pipeline', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frames == 0:
        print('Nenhum quadro processado.', file=sys.stderr)
        return 2
    print(f'Quadros: {frames} | campo detectado em {with_field} de {frames} | '
          f'{1000.0 * elapsed / frames:.1f} ms/quadro' + (f' | saída: {output}' if output else ''))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as error:        # fonte/saída/CSV inválidos: mensagem, não traceback
        print(f'Erro: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
