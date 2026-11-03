# Development

Environments, dependencies, tests, tools and maintenance procedures.

## Where to change things

| I want to… | File |
|---|---|
| change the segmentation algorithm | `src/pipeline/` (no ROS) |
| change the validation rules | `src/Webots_YOLOv8/detection_validator.py` |
| change the class scheme | `detections.py` (`YoloClass`, `EXPECTED_CLASS_NAMES`) + validator + `vision_messages.py` |
| add a parameter | `vision_parameters.py` (declaration + reading) and `config/vision_params.yaml` |
| change what the node publishes | `yolo_simulation.py` and `vision_messages.py` |
| evaluate/train models | `src/Webots_YOLOv8/tools/` |

See [architecture.md](architecture.md) for the flow and the design decisions.

## Reference environments

What was **actually run** to validate the code, and what was **not**:

| Environment | Python | What was verified |
|---|---|---|
| **A.** Numeric stack of Ubuntu 22.04 / Humble (apt): numpy 1.21.5, pandas 1.3.5, OpenCV 4.5.5, pytest 6.2.5 | 3.10 | tests that need neither ROS nor Ultralytics; `setup.py develop` with Humble's setuptools 59.6 |
| **B.** PyTorch 2.14.1 (CPU) + Ultralytics 8.4.184 + ONNX Runtime 1.31.0 | 3.12 | contract tests with real YOLOv8n and YOLO26n; benchmarks; training and comparison tools |
| **C.** ROS 2 Humble via RoboStack (conda-forge), with `soccer_ipm`, `ipm_library` and `soccer_interfaces` built | 3.14 | `colcon build` (normal and `--symlink-install`); full suite with real `rclpy` and real `soccer_ipm` |
| **Target.** Ubuntu 22.04 + ROS 2 Humble (apt) + Webots | 3.10 | **not tested end to end**; nor the node with real weights inside Webots |

> Environment C uses Python 3.14, newer than the Python 3.10 of official Humble; environment A covers Python
> 3.10 only for the non-ROS part. Running the whole suite on the target is the next validation step.

### Reproducing environment A (no ROS, Python 3.10, Ubuntu 22.04 versions)

```bash
python3.10 -m venv .venv-a && source .venv-a/bin/activate
pip install numpy==1.21.5 pandas==1.3.5 opencv-python-headless==4.5.5.64 pytest==6.2.5
python -m pytest test -q          # tests that require ROS or Ultralytics are skipped
```

### Reproducing environment B (detector and tools)

```bash
python3 -m venv .venv-b && source .venv-b/bin/activate
pip install -r requirements.txt ultralytics onnx onnxruntime
python -m pytest test -q
```

`pip install ultralytics` installs PyTorch. It was verified with the conda-forge CPU PyTorch; for a CPU-only
install via pip, use PyTorch's CPU index (not verified here).

## ROS 2 dependencies and the `soccer_ipm` patch

The node publishes to `soccer_ipm`, which is **not** part of this repository. It and its packages come from
[ros-sports](https://github.com/ros-sports) (all three have a `humble` branch):

```bash
cd ~/ws/src
git clone -b humble https://github.com/ros-sports/soccer_interfaces.git
git clone -b humble https://github.com/ros-sports/ipm.git
git clone -b humble https://github.com/ros-sports/soccer_ipm.git
git -C soccer_ipm apply ../robocup-vision-pipeline/docs/patches/soccer_ipm_camera_info.patch
```

**Why the patch.** In `soccer_ipm` (`humble` and `rolling` branches, verified), if a goalpost, a robot or an
obstacle arrives **before** the first `camera_info`, `object_at_bottom_of_image` accesses `camera_info.height`
with `camera_info = None` and raises `AttributeError`. Inside an `rclpy` callback this brings down the
`soccer_ipm` process. It is a real race at system startup. The patch makes the function raise
`CameraInfoNotSetException`, which the existing decorator already handles with a warning. It is 5 lines in
`soccer_ipm/soccer_ipm/utils.py`; ideally it should be proposed upstream.

Without the patch the system works in most startups, but the regression test
(`test_ipm_survives_frames_that_arrive_before_camera_info`) is **skipped** with a message explaining why.

## Build

```bash
source /opt/ros/humble/setup.bash
cd ~/ws
colcon build --packages-select Webots_YOLOv8        # or: colcon build --symlink-install ...
source install/setup.bash
```

Both build modes work. `src/pipeline` and `src/Webots_YOLOv8` live under a single `package_dir` (`src/` layout)
precisely so that `--symlink-install` also works with Humble's old setuptools: with two package root directories,
the legacy `setup.py develop` left `Webots_YOLOv8` off `sys.path`. The `pipeline` and `Webots_YOLOv8` packages are
installed side by side.

Weights placed in `modelo/` **before** the build are installed into `share/Webots_YOLOv8/modelo/`. With
`--symlink-install`, files added afterwards do not show up; use an absolute path in `model_file`.

## Tests

The tests run straight from the source tree: `test/conftest.py` puts `src/` on `sys.path`, so **nothing needs to be
installed** for the non-ROS tests.

```bash
python -m pytest test -q                    # everything the environment allows
python -m pytest test/test_pipeline_cli.py  # one file
```

Modules that depend on missing packages are **skipped**, not failed:

| File | Tests | Covers | Requires |
|---|---|---|---|
| `test_detections.py` | 3 | `Detection` geometry; YOLO output conversion; class names | numpy |
| `test_detection_validator.py` | 9 | accept/reject rules; margin; intersections; crossbar; no field; mask at another scale | OpenCV |
| `test_vision_debug.py` | 3 | accepted/rejected/ignored colors; input frame preserved | OpenCV |
| `test_pipeline_independence.py` | 1 | `pipeline` works **without** ROS, Ultralytics and the rest of the project | OpenCV |
| `test_pipeline_cli.py` | 6 | `python -m pipeline`: image, video, HSV CSV, invalid inputs | OpenCV |
| `test_vision_messages.py` | 6 | real `soccer_vision_2d_msgs` messages (`float` types, empty…) | ROS 2 |
| `test_vision_node.py` | 14 | full node on real `rclpy`; parameters; YAML; launch; `model_file`; classes | ROS 2, installed package |
| `test_end_to_end_ipm.py` | 4 | with the real `soccer_ipm`: distances; `camera_info` regression | ROS 2, `soccer_ipm` |
| `test_running_inference.py` | 10 | contract with the real YOLOv8n and YOLO26n architectures; `load_model` | Ultralytics |
| `test_training_tools.py` | 2 | training recipe read from a reference checkpoint | PyTorch, Ultralytics |

The ROS tests use a **fake YOLO** (a mocked `ultralytics` module) with the real `running_inference.py`,
segmentation, validation and messages, and dummy weights in `tmp_path`: they **do not depend on real weights**.
`test/ros_harness.py` sets up the vision node and a "probe" node that publishes images and listens to the outputs.

The node and end-to-end tests need the package **installed** (they use `get_package_share_directory`):

```bash
colcon build --packages-select Webots_YOLOv8 && source install/setup.bash
python -m pytest test -q
```

Verified count: 58 unique tests (46 pass in environment C, which skips the 12 that require Ultralytics; those 12
pass in environment B). The defects the tests cover were checked by **mutation** (introducing the bug on purpose
and watching the test fail), for example: a validator that accepts everything, `int` in a `double` field, an empty
boundary published, an invalid parameter accepted, `pipeline` importing `rclpy`.

### Style

```bash
pip install flake8
flake8 src test setup.py launch      # the limit (120 columns) is in setup.cfg
```

## Evaluation tools

They run as modules, from the root, with `export PYTHONPATH=$PWD/src`:

```bash
python -m Webots_YOLOv8.tools.compare_models --help    # speed, ONNX, pipeline cost and accuracy
python -m Webots_YOLOv8.tools.train_yolo26n --help     # YOLO26n training recipe identical to the baseline
```

Details and methodology in [benchmarks.md](benchmarks.md).

## Publishing trained weights

"Publishing" here answers two **separate** questions. Do not mix them up:

1. **Do the weights live inside the Git repository, or outside it?**
2. **If outside, which hosting service holds them?**

By default in this project the weights are **kept out of Git** (`.gitignore` excludes them) and the README tells
users where to download them and how to point `model_file` at them. If you decide to version `best.pt` in the
repository instead, remove it from `.gitignore` and update the README accordingly; everything else in this section
still applies, because the license and authorship questions do not go away by committing the file.

### Where to host the file

`best.pt` is small (6.3 MB), so size is **not** the constraint: GitHub only rejects single files above 100 MB.
The real differences are history, citation and licensing.

| Option | Good for | Cost |
|---|---|---|
| **Commit in the repo** | Simplest: `git clone` gives a working model with no extra step | Every retrain adds another ~6 MB binary to the history permanently; the weights become inseparable from the repo's license; no separate hash or model card |
| **GitHub Release asset** | Weights tied to a version tag; repo stays light; stable download URL; you can attach the `sha256` next to it | Lives on GitHub only; not citable on its own |
| **Hugging Face Hub** | Model page with a built-in model card; version history; easy download from code | One more platform and account |
| **Zenodo** | A permanent **DOI**, which fits `CITATION.cff` and scientific reproducibility | Each new version is a new record; slower to update |

Recommendation for this project: a **GitHub Release asset** for day-to-day use, and, if the work will be cited in a
paper or thesis, a **Zenodo** deposit of the same file to get a DOI. Commit the file to the repo only if the license
question below is settled and you accept the history cost.

### Check before publishing, in any of these places

Verify **each** item. The reason for each is in parentheses:

- [ ] **Ultralytics.** The Ultralytics code and models are distributed under AGPL-3.0 (or an enterprise license).
  Weights trained with it are usually treated as subject to those terms; confirm what that requires for your way of
  distributing. (Publishing is the moment this becomes a legal question; using the weights privately is not.)
- [ ] **Dataset.** TORSO-21 declares an MIT license in its `CITATION.cff`, but check the terms of the original
  images and of the dataset converted to YOLO format that was used for training. (Weights derive from the data.)
- [ ] **Authorship.** Who trained the weights and who authorizes redistribution. The `best.pt` checkpoint records
  that it was trained on another computer/user, so confirm with whoever trained it. (You cannot license what you
  do not own.)
- [ ] **Integrity.** Compute `sha256sum best.pt` and publish the hash alongside the file. (Lets anyone confirm they
  downloaded the exact model the benchmarks refer to.)
- [ ] **Model card.** Architecture, classes (table in [architecture.md](architecture.md)), dataset, training
  resolution, metrics and the **usage resolution (320 px)**. (The node refuses a model with a different class
  scheme; the card tells users this before they hit the error.)
- [ ] **README.** Update the "Detector weights" section with the real address and the hash.

### After publishing

Users download the file, check it with `sha256sum -c`, and either place it in `modelo/` before the build or pass an
absolute path: `model_file:=/absolute/path/best.pt` (see the note in [Build](#build) about `--symlink-install`).

## Real Webots captures for the README

The images in `docs/img/` marked `*_synthetic` were generated from a **synthetic scene**. To replace them with real
captures: run the simulator and the node with `show_window:=true`, save the image published on
`processed_image_topic` (for example, with the save button of `rqt_image_view`) and put the file in `docs/img/`,
updating the references in the README.

The synthetic images are reproducible: `test/scene.py::make_scene()` generates the scene;
`Webots_YOLOv8.vision_debug.draw_debug_image` draws the validation; `python -m pipeline scene.png` generates the
segmentation figure.

## Relationship with the segmentation repository

`src/pipeline/` is an **adapted copy** of the code from the
[segmentation repository](https://github.com/diegodemiranda/robocup-segmentation-pipeline-bitbots-inspired)
(commit `f301e4f`, of 2026-10-28, used in the comparison), **not a dependency**: there is no submodule, `pip install`
or import between them. Differences from that commit:

| File | State |
|---|---|
| `field_boundary_detector.py`, `obstacle_detector.py`, `line_detector.py` | identical, except the relative import (`from .color_common import …`) |
| `color_common.py` | **extended**: circular hue, `mask_hsv`, `as_min_max`/`from_min_max`, `color_spaces_from_calibration` |
| `main_pipeline.py` | **rewritten**: `VisionPipeline` class, `PipelineConfig`/`PipelineResult`, rescaling to the original resolution, YOLO fusion, `draw_pipeline_overlay` (the original has `draw_debug_image`) |
| `__init__.py`, `__main__.py` | new here |

Fixes made in one repository **do not reach the other on their own**; syncing is manual. When fixing one of the
three identical filters, apply the same diff to the other repository.

## Version and metadata

When releasing a version, update together: `setup.py` (`version`), `package.xml` (`<version>`) and `CITATION.cff`
(`version`, `date-released`). If the **license** changes, update `LICENSE`, `package.xml` (`<license>`),
`setup.py` (`license`) and `CITATION.cff` (`license`).

## Known issues

- **`numpy` 2 and the apt `cv_bridge` (not verified here).** Humble's `cv_bridge` was built against NumPy 1.x; a
  `pip install ultralytics` that pulls in NumPy 2 may break it. If an `_ARRAY_API` error appears, pin
  `pip install "numpy<2"`.
- **No display.** With `show_window:=true` and no graphical server, OpenCV fails to open the window; use
  `show_window:=false`.
- **`rosdep install`** was not run by me; check that it resolves all the keys in your environment.
