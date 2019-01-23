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

## Usage and Results

Make sure you have the ROS bag file in `auro_calibration/bagfiles` folder. Then you can use the following launch files.

### Play ROSBAG File

This launch file will only play the rosbag record file.

```
roslaunch auro_calibration play_rosbag.launch
```

### Run Camera Calibration

This launch file will play the rosbag record and runs the `camera_calibration` package from ROS.

```
roslaunch auro_calibration camera_calibration.launch
```

23 images were automatically selected by the calibrator and the results are stored in `auro_calibration/calibration_data/camera_calibration`  folder. The following results were obtained:

#### Camera Matrix

```
484.130454    0.000000  457.177461
  0.000000  484.452449  364.861413
  0.000000    0.000000    1.000000
```

#### Distortion Coefficients

```
-0.199619, 0.068964, 0.003371, 0.000296, 0.000000
```

### Update the ROS Bag File

This script will update the camera matrices and the distortion coefficients in the `/sensors/camera/camera_info` topic and creates a new bag file in the same location.

```
rosrun auro_calibration update_camera_info.py <original_file.bag> <calibration_file.yaml>
```

### Display Camera Calibration

This launch file will play the updated rosbag record, run `image_proc` for camera rectification and displays the rectified and unrectified images.

```
roslaunch auro_calibration display_camera_calibration.launch
```

**[YouTube Link for Camera Calibration Demo](https://youtu.be/8FHSmFBTL3U)**
[<img src="https://github.com/heethesh/auro_calibration/blob/master/images/camera_calibration.png?raw=true">](https://youtu.be/8FHSmFBTL3U)

### Calibrate Camera-LiDAR Point Correspondences

This script will perform calibration using the matplotlib GUI to pick correspondences in the camera and the LiDAR frames. You first need to play the rosbag record in another terminal.

```
roslaunch auro_calibration play_rosbag.launch
rosrun auro_calibration calibrate_camera_lidar.py --calibrate
```

Press [ENTER] to launch the GUIs and the pick the corresponding points by selecting the four corner points of checkerboard in both the camera and LiDAR frames. 16 such points were selected for calibration at varying position and depths of the checkerboard. OpenCV's PnP RANSAC method is used to find the rotation and translation transforms between the camera and the LiDAR.

**NOTE: The point files are appended and the extrinsics estimates are calculated and refined continuously using a RANSAC approach.**

The point correspondences are saved as following:
- `auro_calibration/calibration_data/lidar_camera_calibration/img_corners.npy`
- `auro_calibration/calibration_data/lidar_camera_calibration/pcl_corners.npy`

The calibrated extrinsics are saved as following:
- `auro_calibration/calibration_data/lidar_camera_calibration/extrinsics.npz`
    - 'euler' : Euler Angles (RPY)
    - 'R'     : Rotation Matrix
    - 'T'     : Translation Offsets (XYZ)

The following calibrated transforms were obtained:

#### Rotation Matrix
```
-9.16347982e-02  -9.95792677e-01  -8.74577923e-05
 1.88123595e-01  -1.72252569e-02  -9.81994299e-01
 9.77861226e-01  -9.00013023e-02   1.88910532e-01
```

#### Euler Angles (RPY rad)

```
-0.44460865  -1.35998386   2.0240699
```

#### Translation Offsets (XYZ m)

```
-0.14614803  -0.49683771  -0.27546327
```

**[YouTube Link for Camera-LiDAR Calibration GUI Demo](https://youtu.be/FgP8jZ_siJI)**
[<img src="https://github.com/heethesh/auro_calibration/blob/master/images/gui_demo.png?raw=true">](https://youtu.be/FgP8jZ_siJI)

### Display Camera-LiDAR Projection

This launch file will play the updated rosbag record, run `calibrate_camera_lidar.py` in projection mode and display the LiDAR point cloud projected on to the image. A static transform is set up between the `world` and the `velodyne` frame.

```
roslaunch auro_calibration display_camera_lidar_calibration.launch
```

**[YouTube Link for Camera-LiDAR Projection Demo](https://youtu.be/lu2HwMWESj8)**
[<img src="https://github.com/heethesh/auro_calibration/blob/master/images/camera_lidar_calibrated.png?raw=true">](https://youtu.be/lu2HwMWESj8)
