#!/usr/bin/env python3
'''
Particle with robot state and observed landmark state information.
'''

import numpy as np

class Particle:
    def __init__(self,pose,N_particles):
        self.initialize(pose,N_particles)
    
    def initialize(self,pose,N_particles):
        self.x = pose[0]
        self.y = pose[1]
        self.theta = pose[2]
        self.weight = 1.0 / N_particles  # Weight associated with the particle
        self.mean = np.zeros((0, 2))
        self.cov = np.zeros((0, 2, 2))
        self.confidence = np.ones((0,1))
        self.measured_ids = np.zeros((0,1))
        self.should_been_seen = np.zeros((0,1))
        self.was_seen = np.zeros((0,1))


if __name__ == '__main__':
    pass