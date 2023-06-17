#!/usr/bin/env python3
'''
FastSlam 1 with Known Correspondences implementation.
See page 450 from Probablistic Robotics by Sebastian Thrun (Author),
Wolfram Burgard (Author) and Dieter Fox (Author).
'''

import numpy as np
import tf.transformations as tf
from scipy.spatial.transform import Rotation
import copy
import time
from particle import Particle
from motion_model import MotionModel
from measurement_model import MeasurementModel


class FastSLAM1():

    def __init__(self):
        '''
        Initialize models, particles and array of predicted positions
        '''
        # Initialize Motion Model object
        # [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6]
        motion_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2])
        self.motion_model = MotionModel(motion_noise)
        # Initialize Measurement Model object
        Q = np.array([[0.05, 0.01],[0.01, 0.05]])
        self.measurement_model = MeasurementModel(Q)
        # Initial Pose [ x, y, zAxis_rotation]
        initial_pose  = [0,0,0]
        self.odom_x = initial_pose[0]
        self.odom_y = initial_pose[1]
        self.odom_theta = initial_pose[2]
        # Array of N particles
        self.N_particles = 50
        # Initial position
        initial_variance = np.array([0,0,0])
        # Create particles
        self.particles = []
        self.timestamp = 0
        for i in range(self.N_particles):
            # Initialize the particle
            particle=Particle(initial_pose,self.N_particles)
            # Apply Gaussian noise to initial position of the particles
            particle.x = np.random.normal(particle.x, initial_variance[0])
            particle.y = np.random.normal(particle.y, initial_variance[1])
            particle.theta = np.random.normal(particle.theta, initial_variance[2])
            self.particles.append(particle)
        # Initialize the array to keep the average position
        self.predicted_position = np.array([[0,0,0]])
        self.odometry = np.array([[0,0,0]])
        self.measured_ids = np.zeros((0,1))
        self.accepted_landmarks = [33, 32]


    def odometry_update(self, control):
        '''
        Update the estimation of the robot position using the previous position and
        the control action [timestamp v, w]. Also update odometry position.
        '''
        # Initialize time in particles
        if self.timestamp == 0:
            self.timestamp = control[0]
            return
        # Update odometry estimate
        x_t = self.motion_model.sample_real_model_velocity(
            self.timestamp, 
            self.odom_x, 
            self.odom_y, 
            self.odom_theta, 
            control)
        self.odom_x = x_t[0]
        self.odom_y = x_t[1]
        self.odom_theta = x_t[2]
        for particle in self.particles:
            # Update robot state estimate
            x_t = self.motion_model.sample_motion_model_velocity(
                self.timestamp, 
                particle.x, 
                particle.y, 
                particle.theta, 
                control)
            particle.x = x_t[0]
            particle.y = x_t[1]
            particle.theta = x_t[2]
        self.timestamp = control[0]


    def landmarks_update(self,measurements):
        '''
        Update landmark mean and covariance for all landmarks of all particles.
        '''
        # Check if measurements are empty
        if len(measurements.transforms) == 0:
            #print("No aruco markers...")
            return
        # Check if measurements late
        if measurements.header.stamp.to_sec() < self.timestamp:
            print("lost a message...")
            return
        # Loop all the measurements in the fiducial transform array
        for transform in measurements.transforms:
            x = transform.transform.translation.z
            y = transform.transform.translation.x
            bearing = np.arctan2(-y,x)
            range = np.sqrt(x**2 + y**2)
            filtered_measurement = np.array([range,bearing,x,-y])
            # Check if it's a new landmark
            if ~np.any(np.isin(self.measured_ids,transform.fiducial_id)):
            # if  (transform.fiducial_id not in self.measured_ids) and\
            #     (transform.fiducial_id in self.accepted_landmarks):
                # If it's a new landmark, add it
                self.measured_ids = np.append(self.measured_ids,[[transform.fiducial_id]],0)
                for particle in self.particles:
                    index = len(self.measured_ids)-1
                    # Initialize mean and covariance for this landmark
                    self.measurement_model.initialize_landmark(
                        particle,
                        filtered_measurement,
                        index)
            else:
            # elif transform.fiducial_id in self.accepted_landmarks:
                for particle in self.particles:
                    #print(self.measured_ids)
                    index = np.where(self.measured_ids == transform.fiducial_id)[0][0]
                    self.measurement_model.landmark_update(
                        particle,
                        filtered_measurement,
                        index
                    )

        # Normalize all weights
        self.weights_normalization()

        # Resample all particles according to the weights
        self.importance_sampling()


    def weights_normalization(self):
        '''
        Normalize weight in all particles so that the sum = 1.
        '''
        # Compute sum of the weights
        sum = 0.0
        for particle in self.particles:
            sum += particle.weight
        # If sum is too small, equally assign weights to all particles
        if sum < 1e-10:
            for particle in self.particles:
                particle.weight = 1.0 / self.N_particles
            return
        for particle in self.particles:
            particle.weight /= sum


    def importance_sampling(self):
        '''
        Resample all particles through the importance factors.
        '''
        # Construct weights vector
        weights = []
        for particle in self.particles:
            weights.append(particle.weight)
        # Resample all particles according to importance weights
        new_indexes = np.random.choice(
            len(self.particles), 
            len(self.particles),
            replace=True, 
            p=weights
            )
        # Update new particles
        new_particles = []
        for index in new_indexes:
            new_particles.append(copy.deepcopy(self.particles[index]))
        self.particles = new_particles


    def get_predicted_position(self):
        '''
        Calculate the average position of the particles
        '''
        x = 0.0
        y = 0.0
        theta = 0.0
        for particle in self.particles:
            x += particle.x
            y += particle.y
            theta += particle.theta
        x /= self.N_particles
        y /= self.N_particles
        theta /= self.N_particles
        # If the position changed enough, save the new estimate
        if np.linalg.norm(self.predicted_position[-1,0:1] - [x,y]) > 0.1\
        or np.sqrt((self.predicted_position[-1,2] - theta)**2) > 0.08:
            self.predicted_position = np.append(
                self.predicted_position, 
                [[x,y,theta]], 
                axis=0
                )
            self.odometry = np.append(
                self.odometry, 
                [[self.odom_x,self.odom_y,self.odom_theta]], 
                axis=0
                )
    

    def get_plot_data(self):
        '''
        Get data to pass to the plotting process
        '''
        x = [particle.x for particle in self.particles]
        y = [particle.y for particle in self.particles]
        ids = self.measured_ids
        mean = [particle.mean for particle in self.particles]
        cov = [particle.cov for particle in self.particles]
        return self.predicted_position, x, y, ids, mean, cov, self.odometry


if __name__ == "__main__":
    pass
