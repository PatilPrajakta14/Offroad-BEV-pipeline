# Off-Road BEV Prototype

An end-to-end pipeline that teaches an autonomous vehicle to "see" its drivable region in off-road environments. It spans data collection in CARLA, geometric transforms with OpenCV, deep-learning segmentation in PyTorch and real-time orchestration in ROS 2

## Features
- **CARLA Simulation:** Capture front-facing RGB + semantic segmentation streams on off-road maps.

- **BEV Homography:** Convert camera images to bird’s-eye view (BEV) in real time with OpenCV.

- **CNN Segmentation:** Train a lightweight UNet-style model in PyTorch to mask drivable terrain.

- **ROS 2 Pipeline:** Four nodes for acquisition, warping, inference, and visualization.

- **Offline Tools:** Scripts for dataset generation, training, evaluation, and visualization.

## Directory Structure
```
off-road_BEV/                  
├── requirements.txt           
└── ros2_ws/          <= ROS 2 Foxy workspace
    ├── src/
    │   └── bev_pipeline/
    │       ├── launch/
    │       │   └── bringup.launch.py
    │       ├── package.xml
    │       ├── setup.py
    │       └── bev_pipeline/
    │           ├── camera_node.py
    │           ├── bev_node.py
    │           ├── segmentation_node.py
    │           ├── visualization_node.py
    │           ├── homography.py
    │           ├── cnn.py
    │           └── models/
    │               └── bev_model_best_val.pth
```

## Prerequisites
- CARLA Simulator 0.9.13 (server) running on localhost:2000 (or set CARLA_HOST)

- Python 3.8 

- PyTorch, OpenCV, NumPy, CARLA Python API.

- ROS 2 Foxy: for real-time pipeline

## Installation & Setup
### 1. Python Environment
```conda create -n offroad-bev python=3.8 -y
conda activate offroad-bev
pip install -r requirements.txt
```

requirements.txt:
```
numpy
opencv-python
torch
carla==0.9.13
```

### 2. CARLA Server

Download & extract CARLA_0.9.13 (Linux or Windows). Then run:

```bash
# Linux
./CarlaUE4.sh -opengl

# Windows (PowerShell)
.\CarlaUE4.exe
```

Ensure it’s listening on port 2000.

### 1) Data Capture
Spawn a vehicle + dual cameras and record raw frames:
```
python scripts/capture_dataset.py --map Town03 --time 20 --out dataset
```
- --map: CARLA map name (e.g. Town03)

- --time: capture duration in seconds

- --out: root output folder

### 2) Preprocessing
Generate BEV-warped images & binary masks:
```
python scripts/process_masks.py
```
Split into training & validation sets:
```
python scripts/split_bev_dataset.py
```
### 3) Training
Train the UNetTiny segmentation model:
```
python scripts/cnn.py --epochs 10 --batch-size 4 --lr 1e-3
```
Checkpoint saved at scripts/models/bev_model_best_val.pth

### 4) Evaluation & Visualization
Compute metrics on held-out data:
```
python scripts/evaluate.py
```
Visually inspect predictions:
```
python scripts/visualize_preds.py
```
### 5) Real-Time ROS 2 Pipeline
Build & source:
```
cd ros2_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```
Launch all nodes:
```
ros2 launch bev_pipeline bringup.launch.py
```
Nodes & Topics:

- camera_node -> publishes /camera/image_raw

- bev_node -> publishes /camera/bev_image

- segmentation_node -> publishes /camera/bev_mask

- visualization_node -> subscribes to all and shows live BEV+mask overlay

Press q in the display window to quit, and Ctrl+C in the terminal to shut down the ROS graph.

### Author: 
Prajakta Patil

Happy off-roading! 🚙🌲
