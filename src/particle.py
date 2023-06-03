#!/usr/bin/env python3

import numpy as np

class Particle:
    def __init__(self,time,pose,N_particles):
        self.initialize(time,pose,N_particles)
    
    def initialize(self,time,pose,N_particles):
        self.timestamp = time
        self.x = pose[0]
        self.y = pose[1]
        self.theta = pose[2]
        self.weight = 1.0 / N_particles  # Weight associated with the particle
        self.mean = np.zeros((0, 2))
        self.cov = np.zeros((0, 2, 2))
        self.landmark_ids = []
        #b=np.append(b,[[[21,0],[0,21]]],0)


if __name__ == '__main__':
    pass
