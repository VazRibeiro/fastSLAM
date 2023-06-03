#!/usr/bin/env python3

import numpy as np

class MotionModel():
    def __init__(self, motion_noise):
        '''
        Initialize motion model with defined control noise
        '''
        self.motion_noise = motion_noise


    def motion_update(self, particle, control):
        '''
        Conduct motion update for a given particle from current state X_t-1 and
        control U_t.
        '''
        # Manage timestamps
        delta_t = control[0] - particle.timestamp
        particle.timestamp = control[0]

        # Predict position and orientation
        particle.x += control[1] * np.cos(particle.theta) * delta_t
        particle.y += control[1] * np.sin(particle.theta) * delta_t
        particle.theta += control[2] * delta_t

        # Limit θ within [-pi, pi]
        if (particle.theta > np.pi):
            particle.theta -= 2 * np.pi
        elif (particle.theta < -np.pi):
            particle.theta += 2 * np.pi

    def sample_motion_model(self, particle, control):
        '''
        Sample next state X_t from current state X_t-1 and control U_t with
        added motion noise.
        '''
        # Apply Gaussian noise to control input
        u = np.random.normal(control[1], self.motion_noise[0])
        w = np.random.normal(control[2], self.motion_noise[1])
        control_noisy = np.array([control[0], u, w])

        # Motion updated
        self.motion_update(particle, control_noisy)

if __name__ == '__main__':
    pass