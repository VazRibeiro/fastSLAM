#!/usr/bin/env python3

import numpy as np

class Particle:
    def __init__(self,timestamp,pose,N_particles):
        self.initialize(timestamp,pose,N_particles)
    
    def initialize(self,timestamp,pose,N_particles):
        self.pose = pose  # Pose of the particle (x, y, theta)
        self.timestamp = timestamp
        self.x = pose[0]
        self.y = pose[1]
        self.theta = pose[2]
        self.weight = 1.0 / N_particles  # Weight associated with the particle
        self.mean = np.zeros((0, 2))
        self.cov = np.zeros((0, 2, 2))
        self.landmark_ids = []
        #b=np.append(b,[[[21,0],[0,21]]],0)


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
