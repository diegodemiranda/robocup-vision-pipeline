#!/usr/bin/env bash
# Launcher do pipeline de visão no robô.
#   ./robot.sh setup            1x (precisa de internet): dependências, workspace e patch
#   ./robot.sh build            compila o pacote
#   ./robot.sh [run] [args]     compila (se preciso) e executa; args = argumentos do launch
# Exemplos:
#   ./robot.sh run camera_topic:=/usb_cam/image_raw camera_info_topic:=/usb_cam/camera_info
#   MODEL=/home/robo/pesos/best.pt ./robot.sh run show_window:=true
# Variáveis: ROS_DISTRO (humble) | WS (~/robot_ws) | MODEL (modelo/best.pt) | NOBUILD=1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTRO="${ROS_DISTRO:-humble}"
WS="${WS:-$HOME/robot_ws}"
PKG="Webots_YOLOv8"
MODEL="${MODEL:-$REPO/modelo/best.pt}"
PATCH="$REPO/docs/pt/patches/soccer_ipm_camera_info.patch"

die() { echo "ERRO: $*" >&2; exit 1; }
source_ros() {
  [ -f "/opt/ros/$DISTRO/setup.bash" ] || die "ROS 2 $DISTRO não encontrado em /opt/ros/$DISTRO"
  set +u; source "/opt/ros/$DISTRO/setup.bash"; set -u
}
source_ws() { [ -f "$WS/install/setup.bash" ] && { set +u; source "$WS/install/setup.bash"; set -u; } || true; }

cmd_setup() {
  source_ros
  for c in git colcon rosdep pip3; do
    command -v "$c" >/dev/null || die "'$c' ausente. Instale: sudo apt install git python3-pip python3-colcon-common-extensions python3-rosdep ros-$DISTRO-tf2-ros"
  done
  [ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
  rosdep update
  mkdir -p "$WS/src"; cd "$WS/src"
  ln -sfn "$REPO" "$WS/src/robocup-vision-pipeline"
  [ -d soccer_interfaces ] || git clone -b "$DISTRO" https://github.com/ros-sports/soccer_interfaces.git
  [ -d ipm ]               || git clone -b "$DISTRO" https://github.com/ros-sports/ipm.git
  [ -d soccer_ipm ]        || git clone -b "$DISTRO" https://github.com/ros-sports/soccer_ipm.git
  if [ -f "$PATCH" ]; then
    git -C soccer_ipm apply --check "$PATCH" 2>/dev/null && git -C soccer_ipm apply "$PATCH" \
      && echo "patch do soccer_ipm aplicado" || echo "patch do soccer_ipm já aplicado (ou não aplicável)"
  else
    echo "AVISO: $PATCH não existe; soccer_ipm pode cair se chegar detecção antes do camera_info" >&2
  fi
  cd "$WS"
  rosdep install --from-paths src --ignore-src -r -y
  python3 -c "import ultralytics" 2>/dev/null || pip3 install --user ultralytics "numpy<2" \
    || pip3 install --user --break-system-packages ultralytics "numpy<2"
  echo "setup OK. Coloque os pesos em $REPO/modelo/best.pt (ou use MODEL=...) e rode ./robot.sh"
}

cmd_build() {
  source_ros
  [ -d "$WS/src/soccer_ipm" ] || die "workspace ausente: rode './robot.sh setup' primeiro"
  ln -sfn "$REPO" "$WS/src/robocup-vision-pipeline"
  cd "$WS"
  if [ -d install/soccer_ipm ]; then
    colcon build --symlink-install --packages-select "$PKG"
  else
    colcon build --symlink-install
  fi
}

cmd_run() {
  [ "${NOBUILD:-0}" = 1 ] || cmd_build
  source_ros; source_ws
  [ -f "$MODEL" ] || die "pesos não encontrados: $MODEL (defina MODEL=/caminho/best.pt)"
  exec ros2 launch "$PKG" vision_robot.launch.py model_file:="$MODEL" "$@"
}

case "${1:-run}" in
  setup) cmd_setup ;;
  build) cmd_build ;;
  run)   shift || true; cmd_run "$@" ;;
  *)     cmd_run "$@" ;;   # ./robot.sh camera_topic:=... também funciona
esac
