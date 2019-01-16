# Auro Robotics - Camera LIDAR Calibration

## Setup

Run the following to clone the `auro_calibration` package in `ros_workspace/src` directory.

```
cd ~/ros_workspace/src
git clone https://github.com/heethesh/auro_calibration

cd ~/ros_workspace/
catkin_make
source devel/setup.bash
```

## Usage

Make sure you have the ROS bag file in `auro_calibration/bagfiles` folder. Then you can use the following launch files.

### Play ROSBAG File

This launch file will only play the rosbag record file.

```
roslaunch auro_calibration play_rosbag.launch

```

### Run Camera Calibration

This launch file will play the rosbag record, run `image_proc` for camera calibration and display the rectified and unrectified output.

```
roslaunch auro_calibration camera_calibration.launch
```