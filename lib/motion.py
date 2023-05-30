#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Velocity motion model for 2D differential drive robot:
    Robot state: [x, y, θ]
    Control: [u, w].

Author: João Penetra
Email: joao.penetra@tecnico.ulisboa.pt
'''

import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from functools import partial


# Global variables to store the previous velocity time and poses 
previous_velocity_time = None
est_pose_x = [] # Estimated through velocity commands
est_pose_y = []

first_time_called = False
first_pose_x = None
first_pose_y = None
pose_x = [] # Taken from robot Odometry
pose_y = [] 

class Particle():
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.timestamp = 0.0

class MotionModel():
    control = [0, -1, 0, 0]

    def __init__(self, motion_noise):
        '''
        Input:
            motion_noise: [noise_x, noise_y, noise_theta, noise_v, noise_w]
                          (in meters / rad).
        '''
        # Create first instance of the Particle class
        self.particle = Particle(0, 0, 0)

        self.motion_noise = motion_noise
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)
        self.pose_sub = rospy.Subscriber('/pose', Odometry, self.pose_callback)

        # Initialize the timer with the corresponding interruption to work at a constant rate
        self.initialize_timer()
        self.initialize_timer_2()

    def vel_callback(self, cmd_vel):
        '''
        Callback function for the command velocity topic subscriber.

        Input:
            cmd_vel: Velocity message received from the topic.
        '''
        global previous_velocity_time

        # Extract linear and angular velocities from the current velocity message
        v_t = cmd_vel.linear.x # linear velocity
        w_t = cmd_vel.angular.z # angular velocity

        # Define the control array (missing delta_t)
        self.control = [1/30, -1, v_t, w_t]

    def pose_callback(self, pose):
        '''
        Callback function for the pose topic subscriber.

        Input:
            pose: robot's estimate of its position and orientation
        '''
        global first_time_called, first_pose_x, first_pose_y

        if not first_time_called:
            # Run the code only the first time the function is called
            first_pose_x = pose.pose.pose.position.x
            first_pose_y = pose.pose.pose.position.y
            first_time_called = True

        pose_x.append(pose.pose.pose.position.x - first_pose_x)
        pose_y.append(pose.pose.pose.position.y - first_pose_y)


    def initialize_timer_2(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        self.timer = rospy.Timer(rospy.Duration(5), self.timer_callback_2)
        self.h_timerActivate = True

    def timer_callback_2(self, timer):
        '''
        Callback function that runs at a fixed rate of 30Hz

        '''
        # Plot the trajectory
        plt.cla()
        plt.plot(pose_x, pose_y, 'r', label='Odometry',)
        plt.plot(est_pose_x, est_pose_y, 'b', label='Motion model')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Trajectory Comparison')
        plt.grid(True)
        plt.legend() 
        plt.show()


    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """
        timer_callback = partial(self.timer_callback, particle=self.particle)
        self.timer = rospy.Timer(rospy.Duration(1.0 / 30), timer_callback) # First iteration, control at 0
        self.h_timerActivate = True


    def timer_callback(self, timer, particle):
        '''
        Callback function that runs at a fixed rate of 30Hz

        '''

        # At a constant rate of 30Hz, build poses
        self.sample_motion_model(particle,self.control)



    def initialize_particle(self, particle):
        '''
        Add motion noise to the robot state in the given particle object.

        Input:
            particle: Particle() object which has been initialized by first
                      ground truth data.
        Output:
            None.
        '''
        # Apply Gaussian noise to the robot state
        particle.x += np.random.normal(0, self.motion_noise[0])
        particle.y += np.random.normal(0, self.motion_noise[1])
        particle.theta += np.random.normal(0, self.motion_noise[2])

    def motion_update(self, particle, control):
        '''
        Conduct motion update for a given particle from current state X_t-1 and
        control U_t.

        Motion Model (simplified):
        State: [x, y, θ]
        Control: [v, w]
        [x_t, y_t, θ_t] = g(u_t, x_t-1)
        x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        θ_t  =  θ_t-1 + w * delta_t

        Input:
            particle: Particle() object to be updated.
            control: control input U_t.
                     [timestamp, -1, v_t, w_t]
        Output:
            None.
        '''
        control = self.control
        delta_t = control[0] # Alterado para 1/30 neste momento

        # Compute updated [timestamp, x, y, theta]
        particle.x += control[2] * np.cos(particle.theta) * 1/30.0
        particle.y += control[2] * np.sin(particle.theta) * 1/30.0
        particle.theta += control[3] * 1/30.0

        # Limit θ within [-pi, pi]
        if (particle.theta > np.pi):
            particle.theta -= 2 * np.pi
        elif (particle.theta < -np.pi):
            particle.theta += 2 * np.pi

        # print(particle.x)
        est_pose_x.append(particle.x)
        est_pose_y.append(particle.y)



    def sample_motion_model(self, particle, control):
        '''
        Implementation for Fast SLAM 1.0.
        Sample next state X_t from current state X_t-1 and control U_t with
        added motion noise.

        Input:
            particle: Particle() object to be updated.
            control: control input U_t.
                     [timestamp, -1, v_t, w_t]
        Output:
            None.
        '''
        control = self.control
        print('control[2] = '+ str(control[2]))

        # Apply Gaussian noise to control input
        v = np.random.normal(control[2], self.motion_noise[3])
        print('v = '+ str(v))
        w = np.random.normal(control[3], self.motion_noise[4])
        print('w = '+ str(w))
        
        control_noisy = np.array([control[0], control[1], v, w])

        # Motion updated
        self.motion_update(particle, control_noisy)


if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('motion_model_node')

    # Create an instance of the MotionModel class with motion noise values
    #motion_noise = [0.01, 0.01, 0.01, 0.01, 0.01]  # Example motion noise values
    motion_noise = [0, 0, 0, 0, 0]  # Example motion noise values
    motion_model = MotionModel(motion_noise)

    # Spin the ROS node to receive pose messages and trigger the callback
    try:
        rospy.spin()
    except KeyboardInterrupt:
        # Handle Ctrl+C manually if rospy.spin() is interrupted
        print("Termination signal received.")
        rospy.signal_shutdown("Termination signal received.")
        sys.exit(0)
