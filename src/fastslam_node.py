#!/usr/bin/env python3
'''
Implements and runs the FastSlam1 ROS node which interfaces the 
ROS configurations, subscribers and publishers with the FastSlam1
with Known Correspondences algorithm. Runs a second process to 
plot the results.
'''

import rospy
import sys
import tf.transformations as tf
import numpy as np
from nav_msgs.msg import Odometry
from fiducial_msgs.msg import FiducialTransformArray
import time
from fastslam1 import FastSLAM1
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import multiprocessing as mp





import numpy as np
from typing import List
from skimage.measure import LineModelND, ransac
import matplotlib.pyplot as plt


class RansacLineInfo(object):
    """Helper class to manage the information about the RANSAC line."""

    def __init__(self, inlier_points: np.ndarray, model: LineModelND):
        self.inliers = inlier_points  # the inliers that were detected by RANSAC algo
        self.model = model  # The LinearModelND that was a result of RANSAC algo

    @property
    def unitvector(self):
        """The unit vector of the model. This is an array of 2 elements (x, y)"""
        return self.model.params[1]


def extract_lines_using_ransac(map_points: np.ndarray, min_samples: int, max_distance: float,
                               min_inliers_allowed: int, iterations: int) -> List[RansacLineInfo]:
    results: List[RansacLineInfo] = []
    starting_points = map_points.copy()

    for index in range(iterations):
        if len(starting_points) <= min_samples:
            print("No more points available. Terminating search for RANSAC")
            break

        model_robust, inliers = ransac(starting_points, LineModelND, min_samples=min_samples,
                                       residual_threshold=max_distance, max_trials=1000)

        inlier_points = starting_points[inliers]
        if len(inlier_points) < min_inliers_allowed:
            print("Not sufficient inliers found %d, threshold=%d, therefore halting" %
                  (len(inlier_points), min_inliers_allowed))
            break

        starting_points = np.delete(starting_points, inliers, axis=0)
        results.append(RansacLineInfo(inlier_points, model_robust))
        print("Found %d RANSAC lines" % len(results))

    return results


def generate_plottable_points_along_line(model: LineModelND, xmin: float, xmax: float, ymin: float,
                                         ymax: float) -> np.ndarray:
    """
    Computes points along the specified line model
    The visual range is
    between xmin and xmax along X axis
        and
    between ymin and ymax along Y axis
    return shape is [[x1,y1],[x2,y2]]
    """
    unit_vector = model.params[1]
    slope = abs(unit_vector[1] / unit_vector[0])
    x_values = None
    y_values = None
    if slope > 1:
        y_values = np.arange(ymin, ymax, 1)
        x_values = model.predict_x(y_values)
    else:
        x_values = np.arange(xmin, xmax, 1)
        y_values = model.predict_y(x_values)

    np_data_points = np.column_stack((x_values, y_values))
    return np_data_points


def plot_lines_on_map(map_points: np.ndarray, lines: List[RansacLineInfo]):
    plt.scatter(map_points[:, 0], map_points[:, 1], label='Map Points')
    for i, line in enumerate(lines):
        plt.scatter(line.inliers[:, 0], line.inliers[:, 1], label='Inliers {}'.format(i + 1))
        plottable_points = generate_plottable_points_along_line(line.model, map_points[:, 0].min(),
                                                                map_points[:, 0].max(), map_points[:, 1].min(),
                                                                map_points[:, 1].max())
        plt.plot(plottable_points[:, 0], plottable_points[:, 1], label='Line {}'.format(i + 1))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lines Detected by RANSAC')







class FastSlamNode:

    def __init__(self):
        # Initialize some necessary variables here
        self.node_frequency = None
        self.vel_sub = None
        self.fid_sub = None
        #flags
        self.camera_flag = False
        self.odometry_flag = False
        self.main_loop_counter = 0
        self.first_odometry_callback = False
        self.control = [0, 0]
        self.measurements = FiducialTransformArray()
        self.data_association = 'none'
        
        # Initialize the ROS node
        rospy.init_node('fastslam_node')
        rospy.loginfo_once('FastSlam node has started')

        # Load parameters from the parameter server
        self.load_parameters()

        # Initialize the publishers and subscribers
        self.initialize_subscribers()
        self.initialize_publishers()

        # Initialize Algorithm
        self.fastslam = FastSLAM1()
        
        # Initialize the timer
        self.initialize_timer()
        
        # Initialize the data queue
        self.data_queue = mp.Queue()
        # Create a separate process for plotting
        self.plot_process = mp.Process(
            target=self.plot_data_process,
            args=(self.data_queue,))
        self.plot_process.start()


    def load_parameters(self):
        """
        Load the parameters from the configuration server (ROS)
        """
        # Node frequency of operation
        self.node_frequency = rospy.get_param('node_frequency', 100)
        rospy.loginfo('Node Frequency: %s', self.node_frequency)


    def initialize_subscribers(self):
        """
        Initialize the subscribers. 
        """
        self.vel_sub = rospy.Subscriber(
            '/pose', 
            Odometry, 
            self.vel_callback)
        self.fid_sub = rospy.Subscriber(
            '/fiducial_transforms', 
            FiducialTransformArray, 
            self.fid_callback)
        #self.pose_sub = rospy.Subscriber('/pose', Odometry, self.pose_callback)


    def initialize_publishers(self):
        """
        Initialize the publishers.
        """
        pass


    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.node_frequency), 
            self.timer_callback)
        self.h_timerActivate = True


    # Odometry callback
    def vel_callback(self, vel):
        '''
        Callback function for the command velocity topic subscriber.
        '''
        self.odometry_flag = True
        self.first_odometry_callback = True
        # Extract linear and angular velocities from the current velocity message
        time = vel.header.stamp
        time = time.to_sec()
        v = vel.twist.twist.linear.x
        w = vel.twist.twist.angular.z
        self.control = [time,v,w]

    # Aruco markers callback
    def fid_callback(self, fiducial_transforms):
        self.camera_flag = True
        self.measurements = fiducial_transforms


    def plot_data_process(self,data_queue):
        """
        Entry point for the separate process responsible for plotting.
        """
        timer = time.time()
        while True:
            previous_timer = timer
            timer = time.time()
            data = data_queue.get()  # Get data from the queue
            if data['terminate_flag']:
                break
            if data['data']:
                predicted_position, x, y, ids, mean, cov, odometry = data['data']
                # Clear all
                plt.cla()
                # Plot start
                plt.scatter(
                    odometry[:,0][0], 
                    odometry[:,1][0], 
                    s=150, c='blue', marker='*',
                    label="Start position")
                # Plot Robot State Estimate (average position)
                plt.plot(
                    predicted_position[:, 0], 
                    predicted_position[:, 1],
                    'r', 
                    label="Robot State Estimate"
                    )
                #Plot Odometry estimate
                plt.plot(
                    odometry[:, 0], 
                    odometry[:, 1],
                    'orange', 
                    label="Odometry estimate"
                    )         
                # Plot particles
                plt.scatter(x, y, s=5, c='k', alpha=0.5)

                # Plot mean points and covariance ellipses
                for i in range(len(mean[0])):
                    # Plot mean point
                    plt.scatter(
                        mean[0][i, 0], 
                        mean[0][i, 1], 
                        c='b', marker='.')
                    # Plot covariance ellipse
                    eigenvalues, eigenvectors = np.linalg.eig(cov[0][i])
                    angle = np.degrees(
                        np.arctan2(eigenvectors[1, 0], 
                        eigenvectors[0, 0])
                        )
                    ellipse = Ellipse(
                        mean[0][i], 2 * np.sqrt(eigenvalues[0]), 
                        2 * np.sqrt(eigenvalues[1]), 
                        angle=angle, 
                         fill=True,
                         alpha=0.4
                        )
                    plt.gca().add_patch(ellipse)
                    # Add ID as text near the mean point
                    plt.text(
                        mean[0][i, 0], 
                        mean[0][i, 1], 
                        str(int(ids[i][0])), 
                        fontsize=10, 
                        ha='center', 
                        va='bottom'
                        )
                # Plot arrows for robot state
                arrow_length = 0.4  # Length of arrows
                dx = arrow_length * np.cos(predicted_position[-1,2])  # Arrow x-component
                dy = arrow_length * np.sin(predicted_position[-1,2])  # Arrow y-component
                plt.quiver(
                    predicted_position[-1,0],
                    predicted_position[-1,1],
                    dx, 
                    dy, 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1, 
                    color='g', 
                    width=0.005
                    )
                # Plot arrows for odometry
                arrow_length = 0.4  # Length of arrows
                dx = arrow_length * np.cos(odometry[-1,2])  # Arrow x-component
                dy = arrow_length * np.sin(odometry[-1,2])  # Arrow y-component
                plt.quiver(
                    odometry[-1,0],
                    odometry[-1,1],
                    dx, 
                    dy, 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1, 
                    color='grey', 
                    width=0.005
                    )


                min_samples = 5  # RANSAC parameter - The minimum number of data points to fit a model to.
                max_distance = 0.3  # RANSAC parameter - The maximum allowed distance for a point to be classified as an inlier.
                min_inliers_allowed = 6  # Custom parameter - A line is selected only if it has more than these many inliers.
                iterations = 500  # Number of RANSAC iterations

                # Extract lines using RANSAC
                lines = extract_lines_using_ransac(mean[0], min_samples, max_distance, min_inliers_allowed, iterations)

                # Plot the lines on the map
                plot_lines_on_map(mean[0], lines)


                # Plot configurations
                plt.title('Fast SLAM 1.0 with known correspondences')
                plt.legend()
                plt.pause(1e-16)
            #print("plotting time: " + str(timer-previous_timer))
        # Terminate the plot process when the loop breaks
        plt.close()

    ################################################################################
    # Main repeating algorithm
    def timer_callback(self, timer):
        """
        Perform repeating tasks.
        """
        time1 = time.time()
        self.main_loop_counter+=1
        
        if self.first_odometry_callback:
            if self.odometry_flag:
                self.odometry_flag  = False
                # Update particle position
                self.fastslam.odometry_update(self.control)
            
            # Update landmark information
            if self.camera_flag:
                self.camera_flag  = False
                # Update landmark estimation
                self.fastslam.landmarks_update(self.measurements)
        
        # Get average position of the particles
        self.fastslam.get_predicted_position()

        # Plot results
        if ((self.main_loop_counter) % 50 == 0):
            # Put the data and termination flag into the queue
            data = {
            'data': self.fastslam.get_plot_data(),
            'terminate_flag': False
            }
            self.data_queue.put(data)
            self.main_loop_counter = 0

        #print("Algorithm time: " + str(time.time()-time1))
    ################################################################################


def main():
    # Create an instance of the FastSlamNode class
    fastslam_node = FastSlamNode()
    # Access the command-line arguments
    args = rospy.myargv(argv=sys.argv)
    # Check if the required number of arguments is passed
    if len(args) == 2:
        fastslam_node.data_association = args[1]
        rospy.loginfo("Received arguments: data_association = %s", args[1])
    else:
        rospy.logwarn("Invalid number of arguments. Data association set to known.")
        fastslam_node.data_association = 'known'

    rospy.spin()
    # Terminate the plot process when the main script exits
    data = {
        'data': (),
        'terminate_flag': True
    }
    fastslam_node.data_queue.put(data)
    fastslam_node.plot_process.terminate()


if __name__ == '__main__':
    main()