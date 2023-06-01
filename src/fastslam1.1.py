#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import copy
from particle import Particle


class FastSLAM1():
    def __init__(self, motion_model, measurement_model):
        pass

    def initialization(self, N_particles):
        '''
        Initialize robots state, landmark state and all particles.

        Input:
            N_particles: number of particles this SLAM algorithms tracks.
        Output:
            None.
        '''
        # Number of particles and landmarks
        self.N_particles = N_particles
        self.N_landmarks = len(self.landmark_indexes)

        # Robot states: [timestamp, x, y, theta]
        # First state is obtained from ground truth
        self.states = np.array([self.groundtruth_data[0]])

        # Landmark states: [x, y]
        self.landmark_states = np.zeros((self.N_landmarks, 2))

        # Table to record if each landmark has been seen or not
        # [0] - [14] represent for landmark# 6 - 20
        self.landmark_observed = np.full(self.N_landmarks, False)

        # Initial particles
        self.particles = []
        for i in range(N_particles):
            particle = Particle()
            particle.initialization(self.states[0], self.N_particles,
                                    self.N_landmarks)
            self.motion_model.initialize_particle(particle)
            self.particles.append(particle)


    def robot_update(self, control):
        '''
        Update robot pose through sampling motion model for all particles.

        Input:
            control: control input U_t.
                     [timestamp, -1, v_t, w_t]
        Output:
            None.
        '''
        for particle in self.particles:
            self.motion_model.sample_motion_model(particle, control)

    def landmark_update(self, measurement):
        '''
        Update landmark mean and covariance for all landmarks of all particles.
        Based on EKF method.

        Input:
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
        Output:
            None.
        '''
        # Return if the measured object is not a landmark (another robot)
        if not measurement[1] in self.landmark_indexes:
            return

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
                    landmark_update(particle, measurement, landmark_idx)

        # Normalize all weights
        self.weights_normalization()

        # Resample all particles according to the weights
        self.importance_sampling()

    def weights_normalization(self):
        '''
        Normalize weight in all particles so that the sum = 1.

        Input:
            None.
        Output:
            None.
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

        Input:
            None.
        Output:
            None.
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

        Input:
            None.
        Output:
            None.
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

    def plot_data(self):
        '''
        Plot all data through matplotlib.
        Conduct animation as the algorithm runs.

        Input:
            None.
        Output:
            None.
        '''
        # Clear all
        plt.cla()


        plt.title('Fast SLAM 1.0 with known correspondences')
        plt.legend()
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))
        plt.pause(1e-16)


if __name__ == "__main__":
    pass
