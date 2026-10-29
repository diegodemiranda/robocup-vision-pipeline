#!/usr/bin/env python3
"""
train_yolo26n.py

Treina o YOLO26n (versão 2) com a MESMA receita do `best.pt` (YOLOv8n, versão 1), lida do
próprio checkpoint, para que a comparação de acurácia seja justa: mesmo dataset, épocas,
resolução, batch, semente e paciência. O otimizador fica em `auto` nos dois: cada modelo
escolhe o seu (o YOLO26 usa o MuSGD).

Uso:
  python tools/train_yolo26n.py --data /caminho/bitbotstorso_dataset/data.yaml --device 0
  python tools/train_yolo26n.py --data data.yaml --dry-run       # só mostra a receita

Depois do treino, copie `runs/.../weights/best.pt` para `modelo/yolo26n_best.pt` e compare:
  python tools/compare_models.py --models modelo/best.pt modelo/yolo26n_best.pt \
      --data /caminho/bitbotstorso_dataset/data.yaml --split test --imgsz 320
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from ultralytics import YOLO

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Hiperparâmetros copiados do checkpoint de referência.
RECIPE_KEYS = ('epochs', 'patience', 'imgsz', 'batch', 'seed', 'workers', 'single_cls',
               'cos_lr', 'close_mosaic', 'deterministic')


def recipe_from_checkpoint(checkpoint: str) -> Dict[str, Any]:
    """Lê de `train_args` do checkpoint os hiperparâmetros que definem a receita."""
    train_args = torch.load(checkpoint, map_location='cpu', weights_only=False)['train_args']
    return {key: train_args[key] for key in RECIPE_KEYS if train_args.get(key) is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', required=True, help='data.yaml do dataset (o mesmo do best.pt)')
    parser.add_argument('--baseline', default=str(PACKAGE_ROOT / 'modelo' / 'best.pt'),
                        help='checkpoint cuja receita será reproduzida')
    parser.add_argument('--weights', default='yolo26n.pt', help='ponto de partida (pré-treino COCO)')
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs')
    parser.add_argument('--name', default='treinamento_bitbotstorso_dataset_yolo26n')
    parser.add_argument('--dry-run', action='store_true', help='mostra a receita e sai')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = recipe_from_checkpoint(args.baseline)
    print(f'Receita de {args.baseline}:')
    for key, value in recipe.items():
        print(f'  {key} = {value}')
    if args.dry_run:
        return

    YOLO(args.weights).train(
        data=args.data, device=args.device, project=args.project, name=args.name, **recipe)


if __name__ == '__main__':
    main()
