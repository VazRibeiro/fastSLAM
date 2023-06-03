#!/usr/bin/env python3

import rospy
import tf.transformations as tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time
from fastslam1 import FastSLAM1


class FastSlamNode:

    def __init__(self):

        # Initialize some necessary variables here
        self.node_frequency = None
        self.vel_sub = None
        self.pub_pioneer_pose = None
        #flags
        self.odometry_flag = False
        self.camera_flag = False
        
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


    def timer_callback(self, timer):
        """
        Perform repeating tasks.
        """
        if self.odometry_flag:
            self.odometry_flag = False
            self.fastslam_algorithm.odometry_update(self.control)

        if self.camera_flag:
            self.camera_flag  = False

        #Publish results
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


    def vel_callback(self, cmd_vel):
        '''
        Callback function for the command velocity topic subscriber.
        Input:  - cmd_vel: Velocity message received from the topic.
        '''
        self.odometry_flag = True
        # Extract linear and angular velocities from the current velocity message
        u = cmd_vel.linear.x
        w = cmd_vel.angular.z
        self.control = [time.time(),u,w]


def main():

    # Create an instance of the FastSlamNode class
    fastslam_node = FastSlamNode()

    # Spin to keep the script for exiting
    rospy.spin()


if __name__ == '__main__':
    main()