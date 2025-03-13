# ros2-instance-segmentation
ROS2 package doing Instance Segmentation using RealSense and YOLO(yolo11n-seg)

ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30

colcon build --packages-select yolo11_segmentation
source install/setup.bash
ros2 run yolo11_segmentation yolo11seg_node
