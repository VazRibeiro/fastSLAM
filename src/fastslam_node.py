#!/usr/bin/env python3

import rospy
import tf.transformations as tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from fiducial_msgs.msg import FiducialTransformArray
import time
from fastslam1 import FastSLAM1
import matplotlib.pyplot as plt
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
        self.fastslam_algorithm = FastSLAM1()
        
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
        self.node_frequency = rospy.get_param('node_frequency', 30)
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


    def publish_pioneer_pose(self):
        predicted_position = self.fastslam_algorithm.get_predicted_position()
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


    def plot_data_process(self,data_queue):
        """
        Entry point for the separate process responsible for plotting.
        """
        while True:
            data = data_queue.get()  # Get data from the queue

            if data['terminate_flag']:
                break

            if data['data']:
                predicted_position, x_values, y_values = data['data']

                # Process and plot the data here
                # Clear all
                plt.cla()
                # Plot Robot State Estimate (average position)
                plt.plot(predicted_position[:, 0], predicted_position[:, 1],
                        'r', label="Robot State Estimate")
                # Plot particles
                plt.scatter(x_values, y_values,
                            s=5, c='k', alpha=0.5, label="Particles")
                # Plot configurations
                plt.title('Fast SLAM 1.0 with known correspondences')
                plt.legend()
                plt.pause(1e-16)
                #plt.show()

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
        self.fastslam_algorithm.odometry_update([time.time()]+self.control)
        
        # Update landmark information
        if self.camera_flag:
            self.camera_flag  = False

        # Publish results
        self.publish_pioneer_pose()
        # Plot results
        if ((self.main_loop_counter) % 2 == 0):
            # Put the data and termination flag into the queue
            data = {
            'data': self.fastslam_algorithm.get_plot_data(),
            'terminate_flag': False
            }
            self.data_queue.put(data)
            self.main_loop_counter = 0

        time2 = time.time()
        print(time2-time1)
        #print(time1-self.past_time)
        self.past_time = time1
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