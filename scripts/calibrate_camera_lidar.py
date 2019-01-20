#!/usr/bin/env python

'''
Script to find the transformation between the Camera and the LiDAR

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading

# External modules
import numpy as np
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS modules
PKG = 'auro_calibration'
import roslib

roslib.load_manifest(PKG)
import rosbag
import rospy
import ros_numpy
import image_geometry
import message_filters

# import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

# Local python modules
# import transformations

# Global variables
FIRST_TIME = True
CAMERA_MODEL = image_geometry.PinholeCameraModel()
KEY_LOCK = threading.Lock()
PAUSE = False


def handle_keyboard():
    global KEY_LOCK, PAUSE
    while True:
        key = raw_input('Press ENTER to pause and pick points')
        with KEY_LOCK:
            PAUSE = True
        time.sleep(0.5)


def optimizer():
    pass


def extract_points(points):
    global KEY_LOCK, PAUSE

    # Select points within chessboard range
    points = np.asarray(points.tolist())
    inrange = np.where((points[:, 0] > 0) & (points[:, 0] < 2.5) & (np.abs(points[:, 1]) < 2.5) & (points[:, 2] < 2))
    points = points[inrange[0]]
    print('Points Available:', points.shape[0])

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Select Points')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, picker=5)

    # Equalize aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Pick points
    picked = []
    corners = []
    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return
        
        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        print(picked)

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Save corner points
    np.save('corners', corners)

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    # Resume listener
    with KEY_LOCK:
        PAUSE = False


def callback(image, camera_info, velodyne):
    global CAMERA_MODEL, FIRST_TIME, PAUSE

    # rospy.loginfo('In')

    # Setup the pinhole camera model
    if FIRST_TIME:
        FIRST_TIME = False
        rospy.loginfo('Setting up camera model...')
        CAMERA_MODEL.fromCameraInfo(camera_info)

    if PAUSE:
        extract_points(ros_numpy.point_cloud2.pointcloud2_to_array(velodyne))
        # pointcloud2_to_xyz_array

    # rospy.loginfo('Out')


def listener(camera_info, image_color, velodyne_points):
    # Start node
    rospy.init_node('calibrate_camera_lidar', anonymous=True)

    # Subscribe to topics
    info_sub = message_filters.Subscriber(camera_info, CameraInfo)
    image_sub = message_filters.Subscriber(image_color, Image)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    print('Init')
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, info_sub, velodyne_sub], queue_size=5, slop=0.1
    )
    ats.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':

    # Get parameters when starting node from a launch file.
    if len(sys.argv) < 1:
        camera_info = rospy.get_param('camera_info')
        image_color = rospy.get_param('image_color')
        velodyne_points = rospy.get_param('velodyne_points')
    else:
        camera_info = '/sensors/camera/camera_info'
        image_color = '/sensors/camera/image_color'
        velodyne_points = '/sensors/velodyne_points'

    # Start keyboard handler thread
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()

    # Start subscriber
    listener(camera_info, image_color, velodyne_points)
