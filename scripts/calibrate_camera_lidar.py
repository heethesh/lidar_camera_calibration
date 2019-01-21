#!/usr/bin/env python

'''
Script to find the transformation between the Camera and the LiDAR

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules
import numpy as np
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS modules
PKG = 'auro_calibration'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import ros_numpy
import image_geometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

# Local python modules
# import transformations

# Global variables
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
CAMERA_MODEL = image_geometry.PinholeCameraModel()
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'


def handle_keyboard():
    global KEY_LOCK, PAUSE
    key = raw_input('Press [ENTER] to pause and pick points\n')
    with KEY_LOCK: PAUSE = True


def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()


def save_data(data, filename, folder, is_image=False):
    # Empty data
    if not len(data): return

    # Handle filename
    filename = os.path.join(PKG_PATH, os.path.join(folder, filename))
    
    # Create folder
    try:
        os.makedirs(os.path.join(PKG_PATH, folder))
    except OSError:
        if not os.path.isdir(os.path.join(PKG_PATH, folder)): raise

    # Save image
    if is_image:
        cv2.imwrite(filename, data)
        return

    # Save points data
    if os.path.isfile(filename):
        rospy.logwarn('Updating file: %s' % filename)
        data = np.vstack((np.load(filename), data))
    np.save(filename, data)


def extract_points_2D(img_msg, now, rectify=False):
    # Read image using CV bridge
    bridge = CvBridge()
    try:
        img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Rectify image
    if rectify: CAMERA_MODEL.rectifyImage(img, img)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points - %d' % now.secs)
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))
        rospy.loginfo('IMG: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save corner points and image
    rect = '_rect' if rectify else ''
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    save_data(corners, 'img_corners%s.npy' % (rect), CALIB_PATH)
    save_data(img, 'image_color%s-%d.jpg' % (rect, now.secs), 
        os.path.join(CALIB_PATH, 'images'), True)


def extract_points_3D(points, now):
    # Select points within chessboard range
    points = np.asarray(points.tolist())
    inrange = np.where((points[:, 0] > 0) & (points[:, 0] < 2.5) & (np.abs(points[:, 1]) < 2.5) & (points[:, 2] < 2))
    points = points[inrange[0]]
    print(points.shape)
    if points.shape[0] > 5:
        rospy.loginfo('PCL points available: %d', points.shape[0])
    else:
        rospy.logwarn('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Select 3D LiDAR Points - %d' % now.secs, color='white')
    # ax.set_xlabel('X Axis'); ax.set_ylabel('Y Axis'); ax.set_zlabel('Z Axis')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, picker=5)

    # Equalize display aspect ratio for all axes
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
    picked, corners = [], []
    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return
        
        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        rospy.loginfo('PCL: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    # Save corner points
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    save_data(corners, 'pcl_corners.npy', CALIB_PATH)


def calibrate(points2D=None, points3D=None):
    folder = os.path.join(PKG_PATH, CALIB_PATH)
    if points2D is None: points2D = np.load(os.path.join(folder, 'img_corners.npy'))
    if points3D is None: points3D = np.load(os.path.join(folder, 'pcl_corners.npy'))
    assert(points2D.shape[0] == points3D.shape[0])

    # camera_matrix = CAMERA_MODEL.intrinsicMatrix()
    # dist_coeffs = CAMERA_MODEL.distortionCoeffs()

    camera_matrix = np.matrix([[484.130454,   0.,       457.177461],
                                [  0.,       484.452449, 364.861413],
                                [  0.,         0.,         1.      ]])

    dist_coeffs = np.matrix([[-0.199619],
                             [ 0.068964],
                             [ 0.003371],
                             [ 0.000296],
                             [ 0.      ]])

    # print(points2D.shape, points3D.shape)

    success, rotation_vector, translation_vector = cv2.solvePnP(points3D, 
        points2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: rospy.logwarn('Optimization unsuccessful')

    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    print(rotation_matrix)
    euler = euler_from_matrix(rotation_matrix)
    print(euler)
    print(translation_vector)


def reproject():
    folder = os.path.join(PKG_PATH, CALIB_PATH)
    img = cv2.imread(os.path.join(folder, 'images/image_color_rect-1548033467.jpg'))
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    test2D_rect = np.asarray([(487.37662337662334, 476.2900432900433),
                            (486.7051112885865, 349.4897496440848),
                            (272.47545053647707, 345.4729435049827),
                            (275.1533212958784, 471.332869196847),
                            (487.37662337662334, 476.2900432900433)])

    test2D = np.asarray([(485.4177489177489, 482.16666666666663),
                        (487.37662337662334, 343.08658008658006),
                        (256.22943722943717, 341.1277056277056),
                        (254.2705627705627, 476.2900432900433)])

    test3D = np.asarray([(1.4933439493179321, -0.3417896628379822, -0.41048699617385864),
                        (1.455967903137207, -0.3470541834831238, -0.18377897143363953),
                        (1.4777989387512207, 0.19648905098438263, -0.18304775655269623),
                        (1.5002293586730957, 0.1933676302433014, -0.40531063079833984)])

    # ret = CAMERA_MODEL.rectifyPoint(test2D[0])

    R = np.asarray([[ 0.41913896, -0.90665843,  0.04788547],
                     [ 0.17962127,  0.03110673, -0.9832439 ],
                     [ 0.88997681,  0.42071708,  0.17589318]])
    T = np.asarray([-0.86821775, -0.47106544, -0.36172562])

    ext = np.matmul(R, test3D.T).T + T
    
    reproj = []
    for i in range(len(ext)):
        reproj.append(CAMERA_MODEL.project3dToPixel(ext[i]))
    reproj.append(CAMERA_MODEL.project3dToPixel(ext[0]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Reprojection')
    ax.set_axis_off()
    ax.imshow(disp)

    # Draw the line
    temp = np.asarray(reproj)
    ax.plot(temp[:, 0], temp[:, 1], c='#FF0000')
    temp = np.asarray(test2D_rect)
    ax.plot(temp[:, 0], temp[:, 1], c='#00FF00')
    plt.show()


def project_point_cloud(points):
    R = np.asarray([[ 0.41913896, -0.90665843,  0.04788547],
                     [ 0.17962127,  0.03110673, -0.9832439 ],
                     [ 0.88997681,  0.42071708,  0.17589318]])
    T = np.asarray([-0.86821775, -0.47106544, -0.36172562])


def callback(image, camera_info, velodyne):
    global CAMERA_MODEL, FIRST_TIME, PAUSE

    # Setup the pinhole camera model
    if FIRST_TIME:
        FIRST_TIME = False
        rospy.loginfo('Setting up camera model')
        CAMERA_MODEL.fromCameraInfo(camera_info)
        reproject()

    if PAUSE:
        now = rospy.get_rostime()
        point_array = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
        img_p = multiprocessing.Process(target=extract_points_2D, args=[image, now])
        pcl_p = multiprocessing.Process(target=extract_points_3D, args=[point_array, now])
        img_p.start(); pcl_p.start()
        img_p.join(); pcl_p.join()

        img_p = multiprocessing.Process(target=extract_points_2D, args=[image, now, True])
        img_p.start(); img_p.join();

        # Calibrate for existing corresponding points
        calibrate()

        # Resume listener
        with KEY_LOCK: PAUSE = False
        start_keyboard_handler()


def listener(camera_info, image_color, velodyne_points):
    # Start node
    rospy.init_node('calibrate_camera_lidar', anonymous=True)

    # Subscribe to topics
    info_sub = message_filters.Subscriber(camera_info, CameraInfo)
    image_sub = message_filters.Subscriber(image_color, Image)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, info_sub, velodyne_sub], queue_size=5, slop=0.1)
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
    # start_keyboard_handler()

    # Start subscriber
    listener(camera_info, image_color, velodyne_points)

    # test3D = np.asarray([[ 1.568, 0.159, -0.082],
    #                     [ 1.733, 0.194, -0.403],
    #                     [ 1.595, -0.375, -0.378],
    #                     [ 1.542, -0.379, -0.083],
    #                     [ 1.729, -0.173, 0.152],
    #                     [ 3.276, 0.876, -0.178]])

    # test2D = np.asarray([[ 309.0, 315 ],
    #                     [ 304, 433 ],
    #                     [ 491, 436 ],
    #                     [ 490, 321 ],
    #                     [ 426, 286 ],
    #                     [ 253, 401 ]])

    # test2D = np.array([[269.94155844,  66.88528139],
    #    [266.02380952, 211.84199134],
    #    [503.04761905, 219.67748918],
    #    [512.84199134,  70.8030303 ]])

    # test3D = np.array([[ 1.31999838,  0.15539639,  0.35613501],
    #    [ 1.39845812,  0.16888808,  0.12323822],
    #    [ 1.29071867, -0.34059632,  0.11678869],
    #    [ 1.21538198, -0.35212162,  0.33905295]])

    test2D = np.asarray([(485.4177489177489, 482.16666666666663),
                        (487.37662337662334, 343.08658008658006),
                        (256.22943722943717, 341.1277056277056),
                        (254.2705627705627, 476.2900432900433)])

    test3D = np.asarray([(1.4933439493179321, -0.3417896628379822, -0.41048699617385864),
                        (1.455967903137207, -0.3470541834831238, -0.18377897143363953),
                        (1.4777989387512207, 0.19648905098438263, -0.18304775655269623),
                        (1.5002293586730957, 0.1933676302433014, -0.40531063079833984)])
    
    # calibrate(test2D, test3D)
