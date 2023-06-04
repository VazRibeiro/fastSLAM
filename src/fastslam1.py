#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from particle import Particle
from motion_model import MotionModel
from measurement_model import MeasurementModel


class FastSLAM1():
    def __init__(self):
        # Initialize Motion Model object
        # [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6]
        motion_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.motion_model = MotionModel(motion_noise)

        # Initialize Measurement Model object
        Q = np.diagflat(np.array([0.05, 0.02])) ** 2
        self.measurement_model = MeasurementModel(Q)
        
        # Initialize Time
        self.initial_timestamp = time.time()
        # Initial Pose [ x, y, zAxis_rotation]
        initial_pose  = [0,0,0]
        # Array of N particles
        self.N_particles = 100
        # Initial position variance
        initial_variance = np.array([0,0,0])

        self.particles = []
        for i in range(self.N_particles):
            # Initialize the particle
            particle=Particle(self.initial_timestamp,initial_pose,self.N_particles)
            # Apply Gaussian noise to initial position of the particles
            particle.x = np.random.normal(particle.x, initial_variance[0])
            particle.y = np.random.normal(particle.y, initial_variance[1])
            particle.theta = np.random.normal(particle.theta, initial_variance[2])
            self.particles.append(particle)


    # Update the estimation of the robot position using the previous position and
    # the control action [timestamp v, w]
    def odometry_update(self, control):
        for particle in self.particles:
            x_t = self.motion_model.sample_motion_model_velocity(particle, control)
            particle.x = x_t[0]
            particle.y = x_t[1]
            particle.theta = x_t[2]


    def camera_update(self, measurement):
        '''
        Update landmark mean and covariance for all landmarks of all particles.
        Based on EKF method.
        Input:  - measurement: measurement data Z_t. [timestamp, #landmark, range, bearing]
        '''
        for particle in self.particles:
            # Get landmark index
            landmark_idx = self.landmark_indexes[measurement[1]]

            # Initialize landmark by measurement if it is newly observed
            if not particle.lm_ob[landmark_idx]:
                self.measurement_model.\
                    initialize_landmark(particle, measurement,
                                        landmark_idx, 1.0/self.N_landmarks)

            # Update landmark by EKF if it has been observed before
            else:
                self.measurement_model.\
                    camera_update(particle, measurement, landmark_idx)

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
        new_indexes =\
            np.random.choice(len(self.particles), len(self.particles),
                             replace=True, p=weights)

        # Update new particles
        new_particles = []
        for index in new_indexes:
            new_particles.append(copy.deepcopy(self.particles[index]))
        self.particles = new_particles


    def state_update(self):
        '''
        Update the robot and landmark states by taking average among all
        particles.
        '''
        # Robot state
        timestamp = self.particles[0].timestamp
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

        self.states = np.append(self.states,
                                np.array([[timestamp, x, y, theta]]), axis=0)

        # Landmark state
        landmark_states = np.zeros((self.N_landmarks, 2))
        count = np.zeros(self.N_landmarks)
        self.landmark_observed = np.full(self.N_landmarks, False)

        for particle in self.particles:
            for landmark_idx in range(self.N_landmarks):
                if particle.lm_ob[landmark_idx]:
                    landmark_states[landmark_idx] +=\
                        particle.lm_mean[landmark_idx]
                    count[landmark_idx] += 1
                    self.landmark_observed[landmark_idx] = True

        for landmark_idx in range(self.N_landmarks):
            if self.landmark_observed[landmark_idx]:
                landmark_states[landmark_idx] /= count[landmark_idx]

        self.landmark_states = landmark_states


    def get_predicted_position(self):
        timestamp = self.particles[0].timestamp
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

        return np.array([timestamp,x,y,theta])
    
    def plot_data(self):
        '''
        Plot all data through matplotlib.
        Conduct animation as the algorithm runs.
        '''
        # Clear all
        plt.cla()

        # States
        #plt.plot(self.states[:, 1], self.states[:, 2],
         #        'r', label="Robot State Estimate")

        # Create a condition that selects all entries
        condition = np.ones(len(self.particles[1]), dtype=bool)
        plt.scatter(self.particles[condition].x, self.particles[condition].y,
                    s=5, c='k', alpha=0.5, label="Particles")

        plt.title('Fast SLAM 1.0 with known correspondences')
        plt.legend()
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))
        plt.pause(1e-16)


if __name__ == "__main__":
    pass
