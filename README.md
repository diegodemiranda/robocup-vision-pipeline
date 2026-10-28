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
git clone [https://github.com/ivan-josef/Webots_YOLOv8.git](https://github.com/ivan-josef/Webots_YOLOv8.git)
```

### 2. Add the YOLOv8 Model
Download your best.pt model and save it in the model folder of the Webots_YOLOv8 package.

### 3. Build de project 
```bash
cd ~/ros2_ws
colcon build
```

### 4. Setup the Environment 
```bash
source install/setup.bash
```

### 5. Run the pipeline 
open 3 terminals and run 

- Terminal 1 - robot driver and simulation
```bash
ros2 launch my_package robot_launch.py
```

- Terminal 2 - detection node and IPM
```bash
ros2 launch Webots_YOLOv8 vision.launch.py
```

- Terminal 3 - teleop control
```bash
ros2 run my_package keyboard_controller
```

### 6. View Distance Data (IPM)

To visualize the calculated distances (in meters) for detected objects in real-time, use the CLI tool to echo the topics:

first you chan choose de topic

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
