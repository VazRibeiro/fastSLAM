#!/usr/bin/env python3
'''
Implements and runs the FastSlam1 ROS node which interfaces the 
ROS configurations, subscribers and publishers with the FastSlam1
with Known and Unknown Correspondences algorithms. Runs a second 
process to plot the results.
'''

import rospy
import sys
import argparse
import os
import tf.transformations as tf
import numpy as np
from nav_msgs.msg import Odometry
from fiducial_msgs.msg import FiducialTransformArray
import time
from fastslam1 import FastSLAM1
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import multiprocessing as mp


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
        self.control = [0, 0, 0, 0, 0]
        self.measurements = FiducialTransformArray()
        self.data_association = 'none'
        self.resampler = 'none'
        self.rmse = 'none'
        self.reject_ids = False
        
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
        self.node_frequency = rospy.get_param('node_frequency', 50)
        rospy.loginfo('Node Frequency:   %s', self.node_frequency)


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
        # x and y are only used for rms error purposes
        x = vel.pose.pose.position.x
        y = vel.pose.pose.position.y
        # v and w are the velocities used for the motion model
        v = vel.twist.twist.linear.x
        w = vel.twist.twist.angular.z
        self.control = [time,v,w,x,y]
              
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
                    label="Pose Estimate"
                    )
                #Plot Odometry estimate
                plt.plot(
                    odometry[:, 0], 
                    odometry[:, 1],
                    'orange', 
                    label="Odometry"
                    )         
                # Plot particles
                plt.scatter(x, y, s=5, c='k', alpha=0.5)

                # Plot mean points and covariance ellipses
                for i in range(len(mean)):
                    if i==0:
                       # Plot mean point
                        plt.scatter(
                            mean[i, 0], 
                            mean[i, 1], 
                            c='b', marker='.')
                    else:
                        # Plot mean point
                        plt.scatter(
                            mean[i, 0], 
                            mean[i, 1], 
                            c='b', marker='.')
                    # Plot covariance ellipse
                    eigenvalues, eigenvectors = np.linalg.eig(cov[i])
                    angle = np.degrees(
                        np.arctan2(eigenvectors[1, 0], 
                        eigenvectors[0, 0])
                        )
                    if i == 0:
                        ellipse = Ellipse(
                            mean[i], 2 * np.sqrt(5.991*eigenvalues[0]), 
                            2 * np.sqrt(5.991*eigenvalues[1]), 
                            angle=angle, 
                            fill=True,
                            alpha=0.4,
                            label = "95% Confidence"
                            )
                    else:
                        ellipse = Ellipse(
                        mean[i], 2 * np.sqrt(5.991*eigenvalues[0]), 
                        2 * np.sqrt(5.991*eigenvalues[1]), 
                        angle=angle, 
                         fill=True,
                         alpha=0.4
                        )
                    plt.gca().add_patch(ellipse)
                    # Add ID as text near the mean point
                    plt.text(
                        mean[i, 0], 
                        mean[i, 1], 
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
                # Plot configurations
                plt.legend(loc='lower left')
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
                if self.data_association =='known':
                    self.fastslam.landmarks_update_known(
                        self.measurements,
                        self.resampler
                        )
                elif self.data_association =='unknown':
                    self.fastslam.landmarks_update_unknown(
                        self.measurements,
                        self.resampler
                        )
        # Save current data
        self.fastslam.save_data()
        if self.rmse == 'rmse_enabled' and self.data_association!='none':
            self.fastslam.write_data_to_files(
                self.control[3], 
                self.control[4],
                self.data_association
                )
        # Plot results
        if ((self.main_loop_counter) % 40 == 0):
            # Put the data and termination flag into the queue
            data = {
            'data': self.fastslam.get_plot_data(self.data_association),
            'terminate_flag': False
            }
            self.data_queue.put(data)
            self.main_loop_counter = 0

        #print("Algorithm time: " + str(time.time()-time1))
    ################################################################################


def main():
    # Delete output files from previous run
    current_dir = os.path.abspath(__file__)
    current_dir = current_dir.rstrip('fastslam_node.py')
    current_dir = current_dir.rstrip('/src')    
    current_dir = current_dir.rstrip('/fastSLAM')
    filename1 = "simulate_data_for_fastslam/output/points_predict.txt"
    filename2 = "simulate_data_for_fastslam/output/landmarks_predict.txt"
    # Specify the file path
    filename_p = os.path.join(current_dir, filename1)
    filename_l = os.path.join(current_dir, filename2)
    # Check if the path is a file (not a directory)
    if os.path.isfile(filename_p):
        os.remove(filename_p)
    if os.path.isfile(filename_l):
        os.remove(filename_l)

    # Create an instance of the FastSlamNode class
    fastslam_node = FastSlamNode()

    # Create an argument parser
    parser = argparse.ArgumentParser(description='FastSLAM Node')
    # Add arguments
    parser.add_argument('-r', '--rmse', action='store_true', help='Generate data for RMS error.')
    parser.add_argument('-k', '--known', action='store_true', help='Known correspondences.')
    parser.add_argument('-s', '--always_resample', action='store_true', help='Resample at every landmark update.')
    parser.add_argument('-id', '--reject_ids', action='store_true', help='Ignore Aruco ids from a list.')
    # Parse the command-line arguments
    args = parser.parse_args()
    # Check arguments
    if args.rmse:
        rospy.loginfo("RMS error:        Enabled")
        fastslam_node.rmse = 'rmse_enabled'
    else:
        rospy.loginfo("RMS error:        Disabled")
        fastslam_node.rmse = 'rmse_disabled'
    if args.known:
        rospy.loginfo("Data association: Known")
        fastslam_node.data_association = 'known'
    else:
        rospy.loginfo("Data association: Unknown")
        fastslam_node.data_association = 'unknown'
    if args.always_resample:
        rospy.loginfo("Resample method:  Every iteration")
        fastslam_node.resampler = 'simple'
    else:
        rospy.loginfo("Resample method:  Selective Resampling")
        fastslam_node.resampler = 'selective'
    if args.reject_ids:
        rospy.loginfo("Ignore ids:       Enabled")
        fastslam_node.reject_ids = True
    else:
        rospy.loginfo("Ignore ids:       Disabled")
        fastslam_node.reject_ids = False

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