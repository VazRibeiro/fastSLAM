#!/usr/bin/env python3

import numpy as np

class Particle:
    def __init__(self, pose, weight):
        self.pose = pose  # Pose of the particle (x, y, theta)
        self.weight = weight  # Weight associated with the particle
        self.mean = np.zeros((1, 2))
        self.cov = np.zeros((1, 2, 2))
        self.occurrences = np.full(1,False) # Estimates observed by the particle
        self.landmark_id = np.zeros((1, 1))

    def motion_update(self, delta_pose):
        # Perform motion update for the particle based on the given delta pose
        # Update the particle's pose accordingly
        # You may want to consider noise and uncertainty in the motion model
        pass

    def landmark_update(self, observation, landmark_id):
        # Perform measurement update for the particle based on the given observation
        # Update the particle's weight based on the observed landmark and landmark estimation
        # You may want to consider noise and uncertainty in the measurement model
        pass


if __name__ == '__main__':
    pass