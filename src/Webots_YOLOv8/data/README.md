# recursos/

Dados de calibração usados em tempo de execução. Hoje há um arquivo:

## `green_pixels.csv`: calibração do verde do campo

Amostras de pixels do **gramado do mundo Webots do projeto de simulação**, em HSV (escala do OpenCV:
H 0–179, S e V 0–255). 382 linhas, colunas `H,S,V,class` (a coluna `class` é só um rótulo, `verde`).

| Canal | Mínimo | Máximo | Média |
|---|---|---|---|
| H | 55 | 58 | 56,8 |
| S | 154 | 174 | 163,9 |
| V | 128 | 159 | 144,8 |

**Como é usado.** `Webots_YOLOv8.vision_parameters.default_hsv_ranges()` (e, por ele, o nó ROS 2) transforma
essas amostras na faixa HSV do campo: percentis 1 e 99 de cada canal, alargados por uma margem de
(8, 40, 50) em (H, S, V), o que dá `field_min = [47, 115, 79]` e `field_max = [66, 213, 209]`. A faixa
pode ser sobrescrita em tempo de execução (`ros2 param set /vision_node hsv.field_min "[...]"`) ou fixada
em `config/vision_params.yaml`, sem precisar deste arquivo.

**Para a sua câmera/campo.** Este CSV descreve o verde **simulado**; num campo real ou noutra iluminação ele
não serve. Gere o seu: amostre pixels do campo (BGR → HSV com `cv2.cvtColor`) e grave um CSV com o
cabeçalho `H,S,V` (a coluna `class` é opcional). Ou ignore o arquivo e defina as faixas pelo YAML.

**Proveniência.** Estes dados vêm do projeto de simulação original (veja [Licença e proveniência](../README.md#licença-e-proveniência)
no README); confirme com o autor antes de redistribuir fora do projeto.

## Limiares do branco

O branco das linhas e das traves usa S ≤ 43 e V ≥ 213, com o matiz livre (0–179). Esses limiares são
`média ± 2σ` de S e V de amostras de pixels brancos da mesma simulação (S: média 6,4, σ 18,7; V: média 239,6,
σ 12,9); o matiz fica livre porque, com saturação próxima de zero, ele é só ruído de renderização. As
constantes estão em `Webots_YOLOv8/vision_parameters.py` (`WHITE_HSV_MIN`, `WHITE_HSV_MAX`). As amostras
brutas de branco **não** acompanham o repositório.
