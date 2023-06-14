#!/usr/bin/env python3
'''
Implements and runs the FastSlam1 ROS node which interfaces the 
ROS configurations, subscribers and publishers with the FastSlam1
with Known Correspondences algorithm. Runs a second process to 
plot the results.
'''

import rospy
import tf.transformations as tf
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
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
        self.pub_pioneer_pose = None
        #flags
        self.camera_flag = False
        self.main_loop_counter = 0
        self.control = [0, 0]
        self.measurements = FiducialTransformArray()
        self.past_time = 0
        
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
        
        # Initialize the timer with the corresponding interruption to work at a constant rate
        self.initialize_timer()
        
        # Initialize the data queue
        self.data_queue = mp.Queue()
        # Create a separate process for plotting
        self.plot_process = mp.Process(target=self.plot_data_process,args=(self.data_queue,))
        self.plot_process.start()


    def load_parameters(self):
        """
        Load the parameters from the configuration server (ROS)
        """
        # Node frequency of operation
        self.node_frequency = rospy.get_param('node_frequency', 45)
        rospy.loginfo('Node Frequency: %s', self.node_frequency)


    def initialize_subscribers(self):
        """
        Initialize the subscribers. 
        """
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)
        self.fid_sub = rospy.Subscriber('/fiducial_transforms', FiducialTransformArray, self.fid_callback)
        #self.pose_sub = rospy.Subscriber('/pose', Odometry, self.pose_callback)


    def initialize_publishers(self):
        """
        Initialize the publishers.
        """
        self.pub_pioneer_pose = rospy.Publisher('/pioneer_pose', Odometry, queue_size=10)


    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.node_frequency), self.timer_callback)
        self.h_timerActivate = True


    def publish_pioneer_pose(self,predicted_position):
        orientation = tf.quaternion_from_euler(0,0,predicted_position[3])
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(predicted_position[0])
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = predicted_position[1]
        msg.pose.pose.position.y = predicted_position[2]
        msg.pose.pose.position.z = 0
        msg.pose.pose.orientation.x = orientation[0]
        msg.pose.pose.orientation.y = orientation[1]
        msg.pose.pose.orientation.z = orientation[2]
        msg.pose.pose.orientation.w = orientation[3]
        self.pub_pioneer_pose.publish(msg)


    # Odometry callback
    def vel_callback(self, cmd_vel):
        '''
        Callback function for the command velocity topic subscriber.
        '''
        # Extract linear and angular velocities from the current velocity message
        v = cmd_vel.linear.x
        w = cmd_vel.angular.z
        self.control = [v,w]

    # Aruco markers callback
    def fid_callback(self, fiducial_transforms):
        self.camera_flag = True
        self.measurements = fiducial_transforms


    def plot_data_process(self,data_queue):
        """
        Entry point for the separate process responsible for plotting.
        """
        while True:
            data = data_queue.get()  # Get data from the queue
            if data['terminate_flag']:
                break
            if data['data']:
                predicted_position, x, y, ids, mean, cov = data['data']
                # Clear all
                plt.cla()
                
                # Plot Robot State Estimate (average position)
                plt.plot(predicted_position[:, 0], predicted_position[:, 1],
                        'r', label="Robot State Estimate")
                
                # Plot particles
                plt.scatter(x, y, s=5, c='k', alpha=0.5, label="Particles")

                # Plot mean points and covariance ellipses
                for i in range(len(mean[0])):
                    # Plot mean point
                    plt.scatter(mean[0][i, 0], mean[0][i, 1], c='b', marker='o')
                    # Plot covariance ellipse
                    eigenvalues, eigenvectors = np.linalg.eig(cov[0][i])
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    ellipse = Ellipse(mean[0][i], 2 * np.sqrt(eigenvalues[0]), 2 * np.sqrt(eigenvalues[1]), angle=angle, fill=False)
                    plt.gca().add_patch(ellipse)

                    # Add ID as text near the mean point
                    plt.text(mean[0][i, 0], mean[0][i, 1], str(ids[i]), fontsize=8, ha='left', va='center')

                # Plot arrows based on theta
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
                    width=0.005)
                
                # Plot configurations
                plt.title('Fast SLAM 1.0 with known correspondences')
                plt.legend()
                plt.pause(1e-16)
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
        
        # Update particle position
        self.fastslam.odometry_update([time.time()]+self.control)
        
        # Update landmark information
        if self.camera_flag:
            self.camera_flag  = False
            self.fastslam.landmarks_update(self.measurements)
        
        # Get average position of the particles
        predicted_position = self.fastslam.get_predicted_position()

        # Publish results
        self.publish_pioneer_pose(predicted_position)

        # Plot results
        if ((self.main_loop_counter) % 4 == 0):
            # Put the data and termination flag into the queue
            data = {
            'data': self.fastslam.get_plot_data(),
            'terminate_flag': False
            }
            self.data_queue.put(data)
            self.main_loop_counter = 0

        time2 = time.time()
        #print(time2-time1)
    ################################################################################


def main():
    # Create an instance of the FastSlamNode class
    fastslam_node = FastSlamNode()

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