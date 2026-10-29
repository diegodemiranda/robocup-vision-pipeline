<p align="center">
  <b>Selecionar Idioma / Select Language</b><br>
  <a href="#-português">🇧🇷 Português</a> • <a href="#-english">🇺🇸 English</a>
</p>

---

<a id="-português"></a>
# 🇧🇷 Português

# Pipeline de Visão Computacional, Controle de Robô e Estimação de Distância com Webots + ROS 2 + YOLOv8

Este projeto demonstra um pipeline integrado de visão computacional e controle de robôs utilizando a simulação **Webots** e o framework **ROS 2**. Ele permite que um robô virtual (**AUREA**) se movimente, detecte objetos usando **YOLOv8** e estime distâncias no mundo real usando **Mapeamento de Perspectiva Inversa (IPM - Inverse Perspective Mapping)**.

## Recursos

- **Simulação Realista**: Utiliza o simulador de robótica **Webots** para criar um ambiente virtual e um robô móvel, AUREA.
- **Comunicação ROS 2**: Estabelece uma conexão bidirecional entre a simulação no Webots e a rede ROS 2, permitindo o fluxo de dados de sensores e a transmissão de comandos de controle.
- **Detecção de Objetos com YOLOv8**: Um nó ROS 2 dedicado processa o fluxo de vídeo da câmera do robô para detectar objetos (Bolas, Robôs, Traves) e desenhar caixas delimitadoras (*bounding boxes*).
- **Estimação de Distância (IPM)**: Implementa o Mapeamento de Perspectiva Inversa para converter coordenadas 2D em pixels provenientes das detecções em coordenadas 3D relativas (em metros) no sistema de referência do robô.
- **Controle Teleoperado (Teleop)**: Controla a movimentação do robô e o movimento da câmera (*pan*) por meio de comandos de teclado.
- **Visualização em Tempo Real**: A imagem processada com as caixas delimitadoras é publicada em uma janela aberta via OpenCV.

## Tecnologias e Dependências

- **ROS 2 Humble**: Middleware de comunicação robótica.
- **Webots**: Ambiente de simulação robótica.
- **Python 3.10**: Linguagem principal utilizada nos nós.
- **YOLOv8 & Ultralytics**: Framework para detecção de objetos.
- **OpenCV & cv_bridge**: Processamento de imagem e integração entre ROS e OpenCV.
- **IPM (Inverse Perspective Mapping)**: Técnica utilizada para projetar pontos 2D da imagem em um plano 3D no solo.

## Estrutura do Projeto

O projeto é composto pelos seguintes pacotes principais do ROS 2 e suas dependências:

- **`my_package`**
  - Contém o driver do robô para o Webots (`my_robot_driver.py`) e arquivos de configuração.
  - Contém também o nó de controle do robô (`keyboard_teleop.py`).
  - Gerencia a conexão com a simulação.

- **`Webots_YOLOv8`**
  - Contém o nó de visão computacional (`yolo_simulation.py`).
  - Utiliza o YOLOv8 para detecção de objetos.
  - Publica as caixas delimitadoras, imagens processadas e integra-se com a lógica de IPM.

- **`soccer_ipm` & `soccer_interfaces`** (Submódulos)
  - Trata da projeção matemática de pixels para metros e define mensagens personalizadas do ROS.

## Como Executar

### Pré-requisitos
- **Webots** instalado (versão compatível com ROS 2 Humble).
- **ROS 2 Humble** instalado e configurado.
- Um **workspace do ROS 2** (ex: `~/ros2_ws`).

### 1. Clonar o Repositório
Clone o repositório na pasta do seu workspace:
```bash
cd ~/ros2_ws
git clone https://github.com/ivan-josef/Webots_YOLOv8.git
```

### 2. Adicionar o Modelo YOLOv8
Baixe o seu modelo `best.pt` e salve-o na pasta de modelos (`model`) dentro do pacote `Webots_YOLOv8`.

### 3. Compilar o Projeto (Build)
```bash
cd ~/ros2_ws
colcon build
```

### 4. Configurar o Ambiente (Setup)
```bash
source install/setup.bash
```

### 5. Executar o Pipeline
Abra 3 terminais e execute:

- **Terminal 1** - Driver do robô e simulação:
```bash
ros2 launch my_package robot_launch.py
```

- **Terminal 2** - Nó de detecção e IPM:
```bash
ros2 launch Webots_YOLOv8 vision.launch.py
```

- **Terminal 3** - Controle teleoperado:
```bash
ros2 run my_package keyboard_controller
```

### 6. Visualizar Dados de Distância (IPM)

Para visualizar as distâncias calculadas (em metros) para os objetos detectados em tempo real, use a ferramenta de linha de comando para monitorar (*echo*) os tópicos:

Primeiro, você pode listar os tópicos disponíveis:

```bash
ros2 topic list
```

Todos os tópicos que terminam em "relative" contêm medições de distância geradas pelo IPM.

Para a bola:

```bash
ros2 topic echo /balls_relative
```

![](https://github.com/ivan-josef/Webots_YOLOv8/blob/main/image/Screenshot%20from%202025-12-09%2022-13-23.png)

Observe que o robô foi projetado para facilitar o teste de modelos.

## Visão: Detecções do YOLOv8 validadas pelo pipeline de segmentação

Cada *frame* da câmera passa pelas mesmas etapas no `vision_node` (`yolo_simulation.py`):

```
imagem da câmera ─┬─▶ YOLOv8 (running_inference.py) ──────────┐
                  └─▶ segmentação (Webots_YOLOv8/pipeline) ──┴─▶ validação ─▶ tópicos ROS 2 ─▶ soccer_ipm (px → m)
```

Uma detecção do YOLO só é **publicada se estiver dentro da área segmentada** (`detection_validator.py`):

| Classe YOLO | Regra |
|---|---|
| ball, goalpost, robot | ponto de contato com o solo (*foot point* - centro inferior da caixa, o mesmo usado pelo `soccer_ipm`) dentro da região do campo |
| L / T / X intersection | centro da caixa dentro da região do campo **e** pixels de linha branca dentro da caixa |
| crossbar | nunca publicada (sem consumidor) |

A região do campo é a envoltória convexa (*convex hull*) do limite de campo detectado. Se nenhum campo for segmentado, nada é publicado além de arrays vazios. A segmentação também publica os tópicos `field_boundary_in_image` e `obstacles_in_image` (obstáculos = o que a segmentação identificou e o YOLO não explicou).

```bash
ros2 launch Webots_YOLOv8 vision.launch.py                      # configurações padrão
ros2 launch Webots_YOLOv8 vision.launch.py show_window:=false   # modo headless (sem janela)
```

Imagem de depuração (*debug image*, publicada em `processed_image_topic` ou visível na janela do OpenCV): tom azul-claro = área onde as detecções são válidas; caixa verde = aceita; caixa vermelha = rejeitada, com o motivo (`outside_field`, `no_marking`, `no_field`); cinza = classe ignorada.

Ajustes em tempo de execução (padrões em `config/vision_pipeline.yaml`; valores inválidos são rejeitados por completo):
```bash
ros2 param set /vision_node validation.footpoint_margin_px 20
ros2 param set /vision_node validation.enabled false          # publica tudo o que o YOLO detectar
ros2 param set /vision_node hsv.field_min "[48, 100, 70]"     # sempre números inteiros
```

Estrutura de arquivos: `detections.py` (tipos) · `detection_validator.py` (regras) · `vision_messages.py` (mensagens ROS) · `vision_parameters.py` (parâmetros) · `vision_debug.py` (sobreposição visual) · `yolo_simulation.py` (nó, apenas orquestração). Exceto as mensagens e o nó, todo o restante é independente de ROS.

Testes (execute `source install/setup.bash` primeiro para habilitar os testes de ROS; caso contrário, eles serão ignorados):
```bash
python -m pytest src/Webots_YOLOv8/test/test_detections.py src/Webots_YOLOv8/test/test_detection_validator.py \
  src/Webots_YOLOv8/test/test_vision_messages.py src/Webots_YOLOv8/test/test_vision_node.py \
  src/Webots_YOLOv8/test/test_end_to_end_ipm.py
```

Correção incluída em `soccer_ipm/utils.py`: `object_at_bottom_of_image` agora dispara uma exceção `CameraInfoNotSetException` quando nenhuma mensagem de `CameraInfo` chegou ainda. Anteriormente, uma mensagem de trave/robô/obstáculo que chegasse primeiro causava falha no `soccer_ipm` com `AttributeError`.

<p align="right"><a href="#-português">⬆ Voltar ao topo</a></p>

---

<a id="-english"></a>
# 🇺🇸 English

# Computer Vision, Robot Control & Distance Estimation Pipeline with Webots + ROS 2 + YOLOv8

This project demonstrates an integrated computer vision and robot control pipeline using the **Webots** simulation and the **ROS 2** framework. It allows a virtual robot (**AUREA**) to move, detect objects using **YOLOv8**, and estimate real-world distances using **Inverse Perspective Mapping (IPM)**.

## Features

- **Realistic Simulation**: Uses the **Webots** robotics simulator to create a virtual environment and a mobile robot, AUREA.
- **ROS 2 Communication**: Establishes a bidirectional connection between the Webots simulation and the ROS 2 network, enabling sensor data flow and control command transmission.
- **Object Detection with YOLOv8**: A dedicated ROS 2 node processes the robot's camera video stream to detect objects (Balls, Robots, Goalposts) and draw bounding boxes.
- **Distance Estimation (IPM)**: Implements Inverse Perspective Mapping to convert 2D pixel coordinates from detections into relative 3D coordinates (meters) in the robot's frame.
- **Teleop Control**: Controls the robot's movement and camera pan using keyboard commands.
- **Real-Time Visualization**: The processed image with bounding boxes is published to a window opened with OpenCV.

## Technologies and Dependencies

- **ROS 2 Humble**: Robotic communication middleware.
- **Webots**: Robotics simulation environment.
- **Python 3.10**: Main language for the nodes.
- **YOLOv8 & Ultralytics**: Object detection framework.
- **OpenCV & cv_bridge**: Image processing and ROS ↔ OpenCV integration.
- **IPM (Inverse Perspective Mapping)**: Technique used for projecting 2D image points to a 3D ground plane.

## Project Structure

The project consists of the main ROS 2 packages and dependencies:

- **`my_package`**
  - Contains the robot driver for Webots (`my_robot_driver.py`) and configuration files.
  - Also contains the robot control node (`keyboard_teleop.py`).
  - Manages the connection with the simulation.

- **`Webots_YOLOv8`**
  - Contains the computer vision node (`yolo_simulation.py`).
  - Uses YOLOv8 for object detection.
  - Publishes bounding boxes, processed images, and integrates with IPM logic.

- **`soccer_ipm` & `soccer_interfaces`** (Submodules)
  - Handles the mathematical projection from pixels to meters and defines custom ROS messages.

## How to Run

### Prerequisites
- **Webots** installed (version compatible with ROS 2 Humble).
- **ROS 2 Humble** installed and configured.
- A **ROS 2 workspace** (e.g., `~/ros2_ws`).

### 1. Clone the Repository
Clone the repository into the folder of your workspace:
```bash
cd ~/ros2_ws
git clone https://github.com/ivan-josef/Webots_YOLOv8.git
```

### 2. Add the YOLOv8 Model
Download your `best.pt` model and save it in the `model` folder of the `Webots_YOLOv8` package.

### 3. Build the project
```bash
cd ~/ros2_ws
colcon build
```

### 4. Setup the Environment
```bash
source install/setup.bash
```

### 5. Run the pipeline
Open 3 terminals and run:

- **Terminal 1** - robot driver and simulation:
```bash
ros2 launch my_package robot_launch.py
```

- **Terminal 2** - detection node and IPM:
```bash
ros2 launch Webots_YOLOv8 vision.launch.py
```

- **Terminal 3** - teleop control:
```bash
ros2 run my_package keyboard_controller
```

### 6. View Distance Data (IPM)

To visualize the calculated distances (in meters) for detected objects in real-time, use the CLI tool to echo the topics:

First, you can list the available topics:

```bash
ros2 topic list
```

All topics ending in "relative" are measures of distance from the IPM.

For Balls:

```bash
ros2 topic echo /balls_relative
```

![](https://github.com/ivan-josef/Webots_YOLOv8/blob/main/image/Screenshot%20from%202025-12-09%2022-13-23.png)

Note that the robot was designed to facilitate model testing.

## Vision: YOLOv8 detections validated by the segmentation pipeline

Every camera frame goes through the same steps in `vision_node` (`yolo_simulation.py`):

```
camera image ─┬─▶ YOLOv8 (running_inference.py) ──────────┐
              └─▶ segmentation (Webots_YOLOv8/pipeline) ──┴─▶ validation ─▶ ROS 2 topics ─▶ soccer_ipm (px → m)
```

A YOLO detection is **only published if it lies inside the segmented area** (`detection_validator.py`):

| YOLO class | Rule |
|---|---|
| ball, goalpost, robot | foot point (bottom-center of the box, same as `soccer_ipm`) inside the field region |
| L / T / X intersection | box center inside the field region **and** white-line pixels inside the box |
| crossbar | never published (no consumer) |

The field region is the convex hull of the detected field boundary. If no field is segmented, nothing is published except empty arrays. Segmentation also publishes `field_boundary_in_image` and `obstacles_in_image` (obstacles = what segmentation saw and YOLO did not explain).

```bash
ros2 launch Webots_YOLOv8 vision.launch.py                      # defaults
ros2 launch Webots_YOLOv8 vision.launch.py show_window:=false   # headless
```

Debug image (`processed_image_topic` or the OpenCV window): light-blue tint = area where detections are valid; green box = accepted; red box = rejected, with the reason (`outside_field`, `no_marking`, `no_field`); gray = ignored class.

Tuning at runtime (defaults in `config/vision_pipeline.yaml`; invalid values are rejected as a whole):
```bash
ros2 param set /vision_node validation.footpoint_margin_px 20
ros2 param set /vision_node validation.enabled false          # publish everything YOLO sees
ros2 param set /vision_node hsv.field_min "[48, 100, 70]"     # always integers
```

Layout: `detections.py` (types) · `detection_validator.py` (rules) · `vision_messages.py` (ROS messages) · `vision_parameters.py` (parameters) · `vision_debug.py` (overlay) · `yolo_simulation.py` (node, orchestration only). Everything except messages and the node is ROS-free.

Tests (`source install/setup.bash` first to enable the ROS ones; they are skipped otherwise):
```bash
python -m pytest src/Webots_YOLOv8/test/test_detections.py src/Webots_YOLOv8/test/test_detection_validator.py \
  src/Webots_YOLOv8/test/test_vision_messages.py src/Webots_YOLOv8/test/test_vision_node.py \
  src/Webots_YOLOv8/test/test_end_to_end_ipm.py
```

Fix included in `soccer_ipm/utils.py`: `object_at_bottom_of_image` now raises `CameraInfoNotSetException` when no `CameraInfo` has arrived yet. Before, a goalpost/robot/obstacle message arriving first crashed `soccer_ipm` with `AttributeError`.

<p align="right"><a href="#-english">⬆ Back to top</a></p>
