#!/usr/bin/env python3

import numpy as np
import random

class MotionModel():
    def __init__(self, motion_noise):
        '''
        Initialize motion model with defined control noise
        '''
        self.parameters = motion_noise


    # Algorithm for sampling poses from Probabilistic Robotics chapter 5, page 124
    def sample_motion_model_velocity(self,timestamp,x,y,theta,control):
        '''
        Sample next state X_t from current state X_t-1 and control U_t with
        added motion noise.
        '''
        # Avoid zero denominators
        control[2] = control[2] if control[2] != 0 else 0.000001

        # Manage timestamps
        delta_t = control[0] - timestamp
        timestamp = control[0]

        v_est = control[1] + self.sample(self.parameters[0]*control[1]**2 + self.parameters[1]*control[2]**2)
        w_est = control[2] + self.sample(self.parameters[2]*control[1]**2 + self.parameters[3]*control[2]**2)
        gamma_est = self.sample(self.parameters[4]*control[1]**2 + self.parameters[5]*control[2]**2)
        vw_ratio = v_est/w_est
        x_est = x - (vw_ratio)*np.sin(theta) + (vw_ratio)*np.sin(theta + w_est*delta_t)
        y_est = y + (vw_ratio)*np.cos(theta) - (vw_ratio)*np.cos(theta + w_est*delta_t)
        theta_est = theta + w_est*delta_t + gamma_est*delta_t
        
        return [x_est,y_est,theta_est,timestamp]


    # Algorithm for sampling poses assuming a perfect model with no noise
    def sample_real_model_velocity(self,timestamp,x,y,theta,control):
        '''
        Sample next state X_t from current state X_t-1 and control U_t without
        added motion noise.
        '''
        # Avoid zero denominators
        control[2] = control[2] if control[2] != 0 else 0.000001

        # Manage timestamps
        delta_t = control[0] - timestamp
        timestamp = control[0]

        v_est = control[1] 
        w_est = control[2] 
        vw_ratio = v_est/w_est
        x_est = x - (vw_ratio)*np.sin(theta) + (vw_ratio)*np.sin(theta + w_est*delta_t)
        y_est = y + (vw_ratio)*np.cos(theta) - (vw_ratio)*np.cos(theta + w_est*delta_t)
        theta_est = theta + w_est*delta_t
        
        return [x_est,y_est,theta_est]


    # Algorithm to sample from a normal distribution from Probabilistic Robotics chapter 5, page 124
    def sample(self,b):
        summatory = 0
        for _ in range(12):
            summatory += random.uniform(-b, b)
        return summatory

if __name__ == '__main__':
    pass