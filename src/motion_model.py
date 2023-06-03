#!/usr/bin/env python3

import numpy as np
import random

class MotionModel():
    def __init__(self, motion_noise):
        '''
        Initialize motion model with defined control noise
        '''
        self.parameters = motion_noise


    def sample_motion_model_velocity(self, particle, control):
        '''
        Sample next state X_t from current state X_t-1 and control U_t with
        added motion noise.
        '''
        # Avoid zero denominators
        control[2] = control[2] if control[2] != 0 else 0.001

        # Manage timestamps
        delta_t = control[0] - particle.timestamp
        particle.timestamp = control[0]

        v_est = control[1] + self.sample(self.parameters[0]*control[1]**2 + self.parameters[1]*control[2]**2)
        w_est = control[2] + self.sample(self.parameters[2]*control[1]**2 + self.parameters[3]*control[2]**2)
        gamma_est = self.sample(self.parameters[4]*control[1]**2 + self.parameters[5]*control[2]**2)
        x = particle.x - (v_est/w_est)*np.sin(particle.theta) + (v_est/w_est)*np.sin(particle.theta + w_est*delta_t)
        y = particle.y + (v_est/w_est)*np.cos(particle.theta) - (v_est/w_est)*np.cos(particle.theta + w_est*delta_t)
        theta = particle.theta + w_est*delta_t + gamma_est*delta_t
        print(self.sample(self.parameters[0]*control[1]**2))
        return [x,y,theta]


    def sample(self,b):
        summatory = 0
        for _ in range(12):
            summatory += random.uniform(-b, b)
        return summatory

if __name__ == '__main__':
    pass