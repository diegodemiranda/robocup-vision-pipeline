# Arquitetura

Como a imagem da câmera vira detecções **validadas** que o `soccer_ipm` converte em metros.

![Arquitetura](img/architecture.png)

## Dois pacotes, uma direção de dependência

| Pacote | Onde | Papel | Depende de ROS? |
|---|---|---|---|
| `pipeline` | `src/pipeline/` | **Algoritmo** de segmentação por cor (campo, obstáculos, pontos de linha) | **Não.** Só numpy, OpenCV e pandas |
| `Webots_YOLOv8` | `src/Webots_YOLOv8/` | **Integração**: YOLO, validação, parâmetros, mensagens e o nó ROS 2 | Só `yolo_simulation.py` e `vision_messages.py` |

A dependência vai **sempre** de `Webots_YOLOv8` para `pipeline`, nunca o contrário. Isso é o que permite
experimentar a segmentação sozinha (`python -m pipeline foto.png`), sem ROS e sem detector. Um teste
(`test/test_pipeline_independence.py`) bloqueia os imports de `rclpy`, `cv_bridge`, `ultralytics`, `torch` e
`Webots_YOLOv8` e exige que `pipeline` funcione assim; se alguém importar um deles ali, o teste falha.

Módulos de `Webots_YOLOv8`:

| Arquivo | Responsabilidade | Usa ROS |
|---|---|---|
| `detections.py` | Classes do modelo, `Detection`, `Verdict`, motivos de rejeição, checagem de nomes de classe | não |
| `detection_validator.py` | Região do campo e regras de validação | não |
| `vision_parameters.py` | Declaração, leitura e validação dos parâmetros | só tipagem |
| `vision_debug.py` | Imagem de debug (campo, caixas aceitas e rejeitadas) | não |
| `vision_messages.py` | Detecções e segmentação → mensagens `soccer_vision_2d_msgs` | sim |
| `running_inference.py` | Carrega o modelo e roda a inferência | só ao resolver o caminho dos pesos |
| `yolo_simulation.py` | Nó: só **orquestra** captura → YOLO → segmentação → validação → publicação | sim |
| `tools/` | `compare_models.py` (benchmark e acurácia) e `train_yolo26n.py` (receita de treino) | não |

## O que acontece em cada frame

1. A câmera publica `/AUREA/camera_optical_frame/image_color`; o callback converte para BGR (`cv_bridge`).
2. **YOLO.** `running_inference.detect_model` roda o modelo a 320 px (confiança 0,45, até 10 detecções) e
   devolve classes, scores e caixas `[centro_x, centro_y, largura, altura]` em pixels da imagem original;
   `detections_from_yolo` as converte em objetos `Detection`.
3. **Segmentação.** `VisionPipeline.run` reduz o frame para 540 px de largura, acha a fronteira do campo, os
   obstáculos e os pontos de linha, e devolve tudo em coordenadas da imagem original.
4. **Validação.** `DetectionValidator.validate` devolve um `Verdict` por detecção: aceita, ou rejeitada com o motivo.
5. **Publicação.** Só as detecções aceitas viram mensagens. Os arrays de bola, trave, robô e interseção são
   publicados a cada frame, **inclusive vazios**, para o behavior saber que "não há bola" em vez de ficar sem
   informação. A fronteira do campo e os obstáculos também são publicados.
6. **Projeção.** O `soccer_ipm` usa TF e `camera_info` para projetar os pixels no plano do campo e publica
   as posições em metros (`*_relative`).
7. **Debug.** Se houver assinante em `processed_image_topic` ou a janela estiver ligada, desenha-se a imagem de
   debug; caso contrário esse custo (≈7 ms) não é pago.

## A segmentação (`src/pipeline/`)

Implementa o padrão *pipe-and-filter* do artigo do Hamburg Bit-Bots: cada módulo é um filtro independente que
recebe uma imagem e produz uma saída estruturada.

| Módulo | O que faz |
|---|---|
| `color_common.py` | Espaços de cor HSV (`ColorSpaceHSV`; o matiz pode ser **circular**, para o vermelho) e o `ColorDetector`. Deriva a faixa do campo de um CSV de calibração (`color_spaces_from_calibration`) |
| `field_boundary_detector.py` | Varredura por **colunas** (1 a cada `column_step`, padrão 4): de cima para baixo até achar verde; com `head_tilted_up`, de baixo para cima com fechamento morfológico. Depois calcula o **convex hull**, que elimina os "dentes" causados por obstáculos |
| `obstacle_detector.py` | A região entre o convex hull e a fronteira real é candidata a obstáculo se a área passar de `min_obstacle_area` (padrão 150 px²). Classifica por cor predominante: vermelho, azul ou branco (trave) |
| `line_detector.py` | Devolve **pontos** (não linhas) sobre as marcações: sorteia pixels abaixo do topo da fronteira, com mais densidade perto do robô e onde houve detecção no quadro anterior |
| `main_pipeline.py` | `VisionPipeline` orquestra os filtros; `PipelineConfig`/`PipelineResult`; reduz a imagem para `process_width` (padrão 540) e devolve os resultados na resolução original; helpers de fusão com o YOLO e de debug |

Sem campo na imagem (menos de 3 pontos de fronteira), o pipeline devolve `field_detected=False` e nada
mais: evita tratar o branco de paredes como linhas do campo. O custo é de ≈2,5 ms por frame (1080×720, 1
vCPU). O nó ROS desliga os pontos de linha (`n_line_samples=0`: não são calculados nem publicados); a validação
das interseções usa a máscara de marcações diretamente.

**Calibração.** As faixas de cor precisam ser calibradas para a sua câmera e iluminação. No nó, a faixa do campo
vem de `recursos/green_pixels.csv` (veja `recursos/README.md`) e pode ser sobrescrita por `hsv.*_min/max`.

## A validação: a detecção só vale dentro da área segmentada

A **região do campo** é o convex hull da fronteira detectada. Regras (`detection_validator.py`):

| Classe do YOLO | Regra | Motivo da rejeição |
|---|---|---|
| bola, trave, robô | O **ponto de apoio** (centro da base da caixa) deve estar na região do campo, com tolerância de `footpoint_margin_px` (12) | `outside_field` |
| interseção L, T, X | O centro deve estar na região do campo **e** pelo menos `min_marking_ratio` (4%) da caixa deve ter pixels de linha | `outside_field`, `no_marking` |
| travessão (`crossbar`) | Nunca é publicado (não há mensagem) | `ignored_class` |
| qualquer | Se a segmentação não achou campo, nada é validado | `no_field` |

O ponto de apoio é o mesmo critério que o `soccer_ipm` usa (centro + metade da altura da caixa), então o que é
validado é exatamente o ponto que será projetado no chão. A máscara do campo fica na resolução de
processamento (540 px); `FieldRegion` converte os pontos para essa escala em vez de redimensionar a máscara a
cada frame, e consulta uma janela de `±margem` pixels. Os pontos fora da imagem são limitados às bordas.

`validation.enabled:=false` publica tudo o que o YOLO detecta (menos o travessão), como se não houvesse validação.

## Mensagens e tópicos

| Tópico | Tipo | Conteúdo |
|---|---|---|
| `balls_in_image` | `soccer_vision_2d_msgs/BallArray` | bolas validadas (centro e caixa em px) |
| `goal_posts_in_image` | `GoalpostArray` | traves validadas |
| `robots_in_image` | `RobotArray` | robôs validados |
| `markings_in_image` | `MarkingArray` | interseções validadas, com número de raios (L=2, T=3, X=4) |
| `field_boundary_in_image` | `FieldBoundary` | fronteira do campo; **não é publicada vazia** |
| `obstacles_in_image` | `ObstacleArray` | o que a segmentação viu e o YOLO não explicou como bola/trave/robô |
| `processed_image_topic` | `sensor_msgs/Image` | imagem de debug (só desenhada com assinante ou janela) |

O `soccer_ipm` descarta objetos cujo ponto de apoio cai nos 20% inferiores da imagem
(`footpoint_out_of_image_threshold` = 0,8, padrão dele), pois estão parcialmente fora do quadro.

## Parâmetros

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `model_file` | `best.pt` | Pesos: nome em `modelo/` ou **caminho absoluto**. **Só na partida** |
| `strict_class_check` | `true` | Recusa na partida um modelo cujas classes não sejam as do nó. **Só na partida** |
| `show_window` | `true` | Janela do OpenCV (argumento do launch; `false` para rodar sem display) |
| `publish_segmentation` | `true` | Publica fronteira e obstáculos |
| `validation.enabled` | `true` | Liga/desliga a validação |
| `validation.footpoint_margin_px` | `12` | Tolerância do ponto de apoio |
| `validation.min_marking_ratio` | `0.04` | Fração mínima de linha na caixa de uma interseção |
| `pipeline.process_width` | `540` | Largura de processamento da segmentação (0 = sem redução) |
| `pipeline.column_step` | `4` | 1 coluna a cada N na varredura da fronteira |
| `pipeline.min_obstacle_area` | `150` | Área mínima de obstáculo (px²) |
| `pipeline.head_tilted_up` | `false` | Varredura de baixo para cima |
| `hsv.{field,white,red,blue}_{min,max}` | derivado | Faixas HSV (**inteiros**); o vermelho usa matiz circular |

Tudo, exceto `model_file` e `strict_class_check`, pode ser alterado em execução com `ros2 param set`. Uma
alteração inválida (por exemplo, matiz fora de 0–179) é **recusada inteira** e o nó mantém os valores anteriores.
Os valores padrão estão em [`config/vision_params.yaml`](../config/vision_params.yaml).

## Esquema de classes do detector

O nó interpreta os IDs do modelo assim (`detections.YoloClass`):

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Classe | `ball` | `goalpost` | `robot` | `L-Intersection` | `T-Intersection` | `X-Intersection` | `crossbar` |

Um modelo com **outro esquema** (por exemplo, com gol direito/esquerdo ou círculo central) teria os IDs lidos com
o significado trocado. Por isso, na partida, o nó compara `model.names` com esse mapa e **recusa o modelo**,
listando as divergências. `strict_class_check:=false` rebaixa isso a um aviso. Para suportar outro esquema, altere
`YoloClass`, `EXPECTED_CLASS_NAMES`, as regras do validador e o construtor de mensagens.

## Decisões de robustez

- **Fronteira vazia nunca é publicada**: o `soccer_ipm` falha com 0 pontos.
- **Todo número vira `float()` antes de entrar numa mensagem**: o `rclpy` recusa `int` em campo `double`
  (`AssertionError`); stubs de teste não pegam isso, por isso os testes usam mensagens ROS reais.
- **Um frame ruim não derruba o nó**: o callback captura a exceção, registra o traceback (no máximo 1 a cada 5 s)
  e segue; a publicação ocorre **antes** do debug.
- **`soccer_ipm` descarta a mensagem, em vez de cair, se o `camera_info` ainda não chegou**: exige o patch em
  [`docs/patches/`](patches/soccer_ipm_camera_info.patch) (veja [development.md](development.md)).
- **Debug sob demanda**: `detect_model(..., draw=False)` pula o `plot()` do Ultralytics.
- **Parâmetros inválidos são recusados atomicamente.**

## Limitações conhecidas

- **O tópico da câmera é fixo** (`CAMERA_TOPIC` em `yolo_simulation.py`: `/AUREA/camera_optical_frame/image_color`).
  Para outra câmera, use um remap: `ros2 run Webots_YOLOv8 finder --ros-args -r /AUREA/camera_optical_frame/image_color:=/minha/camera`.
  No `launch/vision.launch.py`, os remaps do `soccer_ipm` também apontam para os tópicos do AUREA.
- **Formato de saída único**: as mensagens são `soccer_vision_2d_msgs`. Se o seu behavior consome outro formato,
  é preciso um adaptador (não incluído).
- **Um robô, uma câmera**: não há suporte a várias câmeras nem a distorção de lente (`use_distortion` está `false`).
- **Faixas HSV de exemplo**: `DEFAULT_COLOR_SPACES` (usadas por `python -m pipeline` sem `--hsv-csv`) são só um
  ponto de partida e precisam ser calibradas.
