# ROS2 Instance Segmentation using RealSense and YOLO(yolo11n-seg)

---

### Step 1: Install ROS2 Humble

Follow the [official ROS2 Humble installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

---

### Step 2: Install RealSense SDK and RealSense ROS2 wrapper

Follow the [ROS Wrapper for Intel(R) RealSense(TM) Cameras](https://github.com/greedypelican/realsense-ros2)

```bash
sudo apt update && sudo apt upgrade
```
```bash
sudo apt install ros-humble-librealsense2* ros-humble-realsense2-*
```

---

### Step 3: Set Up YOLO11 Segmentation Package

Move the `yolo11_segmentation` directory into the `src` directory of your ROS2 workspace

```bash
git clone https://github.com/greedypelican/realsense-ros2.git
```
```bash
mv ~/realsense-ros2/yolo11_segmentation ~/ros2_ws/src/
```

---

### Step 4: Launch RealSense Camera Node

Open a new terminal and run

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_rgbd:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30
```

---

### Step 5: Build and Run YOLO11 Segmentation Node

Open a new terminal and run

```bash
cd ~/ros2_ws
```
```bash
colcon build --packages-select yolo11_segmentation
```
```bash
source install/setup.bash
```
```bash
ros2 run yolo11_segmentation yolo11seg_node
```

---

### Step 6: Visualize Results in RViz2

Open a new terminal and run

```bash
rviz2
```

Inside RViz:
- Click `Add` → `By topic`
- Select `Image` under `/yolo11/segmentation` to visualize the instance segmentation results.

