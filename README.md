# ros2-instance-segmentation
ROS2 package doing Instance Segmentation using RealSense and YOLO(yolo11n-seg)

<step 1> 
  install ros2-humble ( https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html )

<step 2>
  install realsense sdk ( sudo apt install ros-humble-realsense2-* && sudo apt install ros-humble-realsense2-* )

<step 3>
  move the yolo11_segmentation directory to src directory of your ros2 workspace 


<step 4>
  open first terminal
  ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30

<step 5>
  open second terminal
  colcon build --packages-select yolo11_segmentation
  source install/setup.bash
  ros2 run yolo11_segmentation yolo11seg_node

<step 6>
  open third terminal
  rviz2 ( add -> by topic -> yolo11/segmentation )
