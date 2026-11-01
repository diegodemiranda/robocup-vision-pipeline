# Desenvolvimento

Ambientes, dependências, testes, ferramentas e procedimentos de manutenção.

## Onde mexer

| Quero… | Arquivo |
|---|---|
| mudar o algoritmo de segmentação | `src/pipeline/` (sem ROS) |
| mudar as regras de validação | `src/Webots_YOLOv8/detection_validator.py` |
| trocar o esquema de classes | `detections.py` (`YoloClass`, `EXPECTED_CLASS_NAMES`) + validador + `vision_messages.py` |
| adicionar um parâmetro | `vision_parameters.py` (declaração + leitura) e `config/vision_params.yaml` |
| mudar o que o nó publica | `yolo_simulation.py` e `vision_messages.py` |
| avaliar/treinar modelos | `src/Webots_YOLOv8/tools/` |

Veja [architecture.md](architecture.md) para o fluxo e as decisões.

## Ambientes de referência

O que foi **realmente executado** para validar o código, e o que **não** foi:

| Ambiente | Python | O que foi verificado |
|---|---|---|
| **A.** Pilha numérica do Ubuntu 22.04 / Humble (apt): numpy 1.21.5, pandas 1.3.5, OpenCV 4.5.5, pytest 6.2.5 | 3.10 | testes que não precisam de ROS nem de Ultralytics; `setup.py develop` com o setuptools 59.6 do Humble |
| **B.** PyTorch 2.14.1 (CPU) + Ultralytics 8.4.184 + ONNX Runtime 1.31.0 | 3.12 | testes de contrato com YOLOv8n e YOLO26n reais; benchmarks; ferramentas de treino e comparação |
| **C.** ROS 2 Humble via RoboStack (conda-forge), com `soccer_ipm`, `ipm_library` e `soccer_interfaces` compilados | 3.14 | `colcon build` (normal e `--symlink-install`); suíte completa com `rclpy` real e `soccer_ipm` real |
| **Alvo.** Ubuntu 22.04 + ROS 2 Humble (apt) + Webots | 3.10 | **não testado de ponta a ponta**; nem o nó com pesos reais dentro do Webots |

> O ambiente C usa Python 3.14, mais novo que o Python 3.10 do Humble oficial; o ambiente A cobre o Python
> 3.10 só para a parte sem ROS. Rodar a suíte inteira no alvo é o próximo passo de validação.

### Reproduzir o ambiente A (sem ROS, Python 3.10, versões do Ubuntu 22.04)

```bash
python3.10 -m venv .venv-a && source .venv-a/bin/activate
pip install numpy==1.21.5 pandas==1.3.5 opencv-python-headless==4.5.5.64 pytest==6.2.5
python -m pytest test -q          # testes que exigem ROS ou Ultralytics são pulados
```

### Reproduzir o ambiente B (detector e ferramentas)

```bash
python3 -m venv .venv-b && source .venv-b/bin/activate
pip install -r requirements.txt ultralytics onnx onnxruntime
python -m pytest test -q
```

O `pip install ultralytics` instala o PyTorch. Foi verificado com o PyTorch de CPU do conda-forge; para uma
instalação só de CPU via pip, use o índice de CPU do PyTorch (não verificado aqui).

## Dependências ROS 2 e o patch do `soccer_ipm`

O nó publica para o `soccer_ipm`, que **não** faz parte deste repositório. Ele e seus pacotes vêm do
[ros-sports](https://github.com/ros-sports) (os três têm branch `humble`):

```bash
cd ~/ws/src
git clone -b humble https://github.com/ros-sports/soccer_interfaces.git
git clone -b humble https://github.com/ros-sports/ipm.git
git clone -b humble https://github.com/ros-sports/soccer_ipm.git
git -C soccer_ipm apply ../robocup-vision-pipeline/docs/patches/soccer_ipm_camera_info.patch
```

**Por que o patch.** No `soccer_ipm` (branches `humble` e `rolling`, verificados), se uma trave, um robô ou
um obstáculo chega **antes** do primeiro `camera_info`, `object_at_bottom_of_image` acessa `camera_info.height`
com `camera_info = None` e levanta `AttributeError`. Dentro de um callback do `rclpy` isso derruba o processo
do `soccer_ipm`. É uma corrida real na partida do sistema. O patch faz a função levantar
`CameraInfoNotSetException`, que o decorator existente já trata com um aviso. São 5 linhas em
`soccer_ipm/soccer_ipm/utils.py`; o ideal é propô-lo ao projeto de origem.

Sem o patch o sistema funciona na maioria das partidas, mas o teste de regressão
(`test_ipm_survives_frames_that_arrive_before_camera_info`) é **pulado** com uma mensagem explicando o motivo.

## Build

```bash
source /opt/ros/humble/setup.bash
cd ~/ws
colcon build --packages-select Webots_YOLOv8        # ou: colcon build --symlink-install ...
source install/setup.bash
```

Os dois modos de build funcionam. `src/pipeline` e `src/Webots_YOLOv8` ficam sob um único `package_dir`
(layout `src/`) justamente para o `--symlink-install` funcionar também com o setuptools antigo do Humble:
com dois diretórios-raiz de pacotes, o `setup.py develop` legado deixava `Webots_YOLOv8` fora do
`sys.path`. Os pacotes `pipeline` e `Webots_YOLOv8` são instalados lado a lado.

Pesos em `modelo/` **antes** do build são instalados em `share/Webots_YOLOv8/modelo/`. Com
`--symlink-install`, arquivos adicionados depois não aparecem; use um caminho absoluto em `model_file`.

## Testes

Os testes rodam direto da árvore de fontes: `test/conftest.py` põe `src/` no `sys.path`, então **não é
preciso instalar nada** para os testes sem ROS.

```bash
python -m pytest test -q                    # tudo o que o ambiente permitir
python -m pytest test/test_pipeline_cli.py  # um arquivo
```

Os módulos que dependem de pacotes ausentes são **pulados**, não falham:

| Arquivo | Testes | Cobre | Exige |
|---|---|---|---|
| `test_detections.py` | 3 | geometria de `Detection`; conversão da saída do YOLO; nomes de classe | numpy |
| `test_detection_validator.py` | 9 | regras de aceitação/rejeição; margem; interseções; travessão; sem campo; máscara em outra escala | OpenCV |
| `test_vision_debug.py` | 3 | cores de aceita/rejeitada/ignorada; frame de entrada preservado | OpenCV |
| `test_pipeline_independence.py` | 1 | `pipeline` funciona **sem** ROS, Ultralytics e o resto do projeto | OpenCV |
| `test_pipeline_cli.py` | 6 | `python -m pipeline`: imagem, vídeo, CSV HSV, entradas inválidas | OpenCV |
| `test_vision_messages.py` | 6 | mensagens `soccer_vision_2d_msgs` reais (tipos `float`, vazias…) | ROS 2 |
| `test_vision_node.py` | 14 | nó completo em `rclpy` real; parâmetros; YAML; launch; `model_file`; classes | ROS 2, pacote instalado |
| `test_end_to_end_ipm.py` | 4 | com o `soccer_ipm` real: distâncias; regressão do `camera_info` | ROS 2, `soccer_ipm` |
| `test_running_inference.py` | 10 | contrato com as arquiteturas reais YOLOv8n e YOLO26n; `load_model` | Ultralytics |
| `test_training_tools.py` | 2 | receita de treino lida de um checkpoint de referência | PyTorch, Ultralytics |

Os testes de ROS usam um **YOLO falso** (módulo `ultralytics` simulado) com o `running_inference.py`, a
segmentação, a validação e as mensagens reais, e pesos de mentira em `tmp_path`: **não dependem de pesos
reais**. `test/ros_harness.py` monta o nó de visão e um nó "probe" que publica imagens e escuta as saídas.

Os testes de nó e de ponta a ponta precisam do pacote **instalado** (usam `get_package_share_directory`):

```bash
colcon build --packages-select Webots_YOLOv8 && source install/setup.bash
python -m pytest test -q
```

Contagem verificada: 58 testes únicos (46 passam no ambiente C, que pula os 12 que exigem Ultralytics; esses
12 passam no ambiente B). Os defeitos que os testes cobrem foram conferidos por **mutação** (introduzir o erro
de propósito e ver o teste falhar), por exemplo: validador que aceita tudo, `int` em campo `double`,
fronteira vazia publicada, parâmetro inválido aceito, `pipeline` importando `rclpy`.

### Estilo

```bash
pip install flake8
flake8 src test setup.py launch      # o limite (120 colunas) está em setup.cfg
```

## Ferramentas de avaliação

Rodam como módulos, a partir da raiz, com `export PYTHONPATH=$PWD/src`:

```bash
python -m Webots_YOLOv8.tools.compare_models --help    # velocidade, ONNX, custo do pipeline e acurácia
python -m Webots_YOLOv8.tools.train_yolo26n --help     # receita de treino do YOLO26n igual à do baseline
```

Detalhes e a metodologia em [benchmarks.md](benchmarks.md).

## Publicar pesos treinados

Os pesos **não** ficam no repositório (`.gitignore` os exclui). Antes de publicá-los em qualquer lugar (por
exemplo, como *asset* de um GitHub Release), verifique **cada** item:

- [ ] **Ultralytics.** O código e os modelos do Ultralytics são distribuídos sob AGPL-3.0 (ou licença
  empresarial). Pesos treinados com ele costumam ser tratados como sujeitos a esses termos; confirme o que isso
  exige na sua forma de distribuição.
- [ ] **Dataset.** O TORSO-21 declara licença MIT no `CITATION.cff` dele, mas confira os termos das imagens
  originais e do dataset convertido para o formato YOLO que foi usado no treino.
- [ ] **Autoria.** Quem treinou os pesos e quem autoriza a redistribuição. O checkpoint do `best.pt` registra
  que foi treinado em outro computador/usuário, então confirme com quem o treinou.
- [ ] **Integridade.** Calcule `sha256sum best.pt` e publique o hash junto do arquivo.
- [ ] **Model card.** Arquitetura, classes (tabela do [architecture.md](architecture.md)), dataset, resolução
  de treino, métricas e a **resolução de uso (320 px)**.
- [ ] **README.** Atualize a seção "Pesos do detector" com o endereço real e o hash.

## Capturas reais do Webots para o README

As imagens em `docs/img/` marcadas como `*_synthetic` foram geradas de uma **cena sintética**. Para substituí-las
por capturas reais: rode o simulador e o nó com `show_window:=true`, salve a imagem publicada em
`processed_image_topic` (por exemplo, pelo botão de salvar do `rqt_image_view`) e coloque o arquivo em
`docs/img/`, atualizando as referências no README.

As imagens sintéticas são reproduzíveis: `test/scene.py::make_scene()` gera a cena;
`Webots_YOLOv8.vision_debug.draw_debug_image` desenha a validação; `python -m pipeline cena.png` gera a
figura de segmentação.

## Relação com o repositório de segmentação

`src/pipeline/` é uma **cópia adaptada** do código do
[repositório de segmentação](https://github.com/diegodemiranda/robocup-segmentation-pipeline-bitbots-inspired)
(commit `f301e4f`, de 28/10/2026, na comparação), **não uma dependência**: não há submódulo, `pip install` nem
import entre eles. Diferenças em relação a esse commit:

| Arquivo | Estado |
|---|---|
| `field_boundary_detector.py`, `obstacle_detector.py`, `line_detector.py` | idênticos, exceto o import relativo (`from .color_common import …`) |
| `color_common.py` | **estendido**: matiz circular, `mask_hsv`, `as_min_max`/`from_min_max`, `color_spaces_from_calibration` |
| `main_pipeline.py` | **reescrito**: classe `VisionPipeline`, `PipelineConfig`/`PipelineResult`, reescala para a resolução original, fusão com o YOLO, `draw_pipeline_overlay` (o original tem `draw_debug_image`) |
| `__init__.py`, `__main__.py` | novos aqui |

Correções feitas em um repositório **não chegam sozinhas** ao outro; a sincronia é manual. Ao corrigir um dos três
filtros idênticos, aplique o mesmo diff no outro repositório.

## Versão e metadados

Ao lançar uma versão, atualize juntos: `setup.py` (`version`), `package.xml` (`<version>`) e `CITATION.cff`
(`version`, `date-released`). Se a **licença** mudar, atualize `LICENSE`, `package.xml` (`<license>`),
`setup.py` (`license`) e `CITATION.cff` (`license`).

## Problemas conhecidos

- **`numpy` 2 e o `cv_bridge` do apt (não verificado aqui).** O `cv_bridge` do Humble foi compilado com NumPy
  1.x; um `pip install ultralytics` que traga NumPy 2 pode quebrá-lo. Se aparecer erro de `_ARRAY_API`, fixe
  `pip install "numpy<2"`.
- **Sem display.** Com `show_window:=true` e sem servidor gráfico, o OpenCV falha ao abrir a janela; use
  `show_window:=false`.
- **`rosdep install`** não foi executado por mim; confira se resolve todas as chaves no seu ambiente.
