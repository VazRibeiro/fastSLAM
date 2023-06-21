#!/usr/bin/env python3

import numpy as np

class MeasurementModel():
    def __init__(self, Q): #recebe uma matriz de covariância
        Q = ([[0.0, 0.0],
              [0.0, 0.0]]) 
        self.Q = Q

    def compute_expected_measurement(self, particle, landmark_idx):
        delta_x = particle.lm_mean[landmark_idx, 0] - particle.x
        delta_y = particle.lm_mean[landmark_idx, 1] - particle.y
        q = delta_x ** 2 + delta_y ** 2

        range = np.sqrt(q)
        bearing = np.arctan2(delta_y, delta_x) - particle.theta

        return range, bearing

    def compute_expected_landmark_state(self, particle, measurement):
        x = particle.x + measurement[2] *\
            np.cos(measurement[3] + particle.theta)
        y = particle.y + measurement[2] *\
            np.sin(measurement[3] + particle.theta)

        return x, y

    def compute_landmark_jacobian(self, particle, landmark_idx):

        delta_x = particle.lm_mean[landmark_idx, 0] - particle.x
        delta_y = particle.lm_mean[landmark_idx, 1] - particle.y
        q = delta_x ** 2 + delta_y ** 2

        H_1 = np.array([delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
        H_2 = np.array([-delta_y/q, delta_x/q])
        H_m = np.array([H_1, H_2])

        return H_m

    def initialize_landmark(self, particle, measurement, landmark_idx, weight):
        # Update landmark mean by inverse measurement model
        particle.lm_mean[landmark_idx] =\
            self.compute_expected_landmark_state(particle, measurement)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle, landmark_idx)

        # Update landmark covariance
        H_inverse = np.linalg.inv(H_m)
        particle.lm_cov[landmark_idx] = H_inverse.dot(self.Q).dot(H_inverse.T)

        # Mark landmark as observed
        particle.lm_ob[landmark_idx] = True

        # Assign default importance weight
        particle.weight = weight

        # Update timestamp
        particle.timestamp = measurement[0]

    def landmark_update(self, particle, measurement, landmark_idx):
        # Compute expected measurement
        range_expected, bearing_expected =\
            self.compute_expected_measurement(particle, landmark_idx)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle, landmark_idx)

        # Compute Kalman gain
        Q = H_m.dot(particle.lm_cov[landmark_idx]).dot(H_m.T) + self.Q
        K = particle.lm_cov[landmark_idx].dot(H_m.T).dot(np.linalg.inv(Q))

        # Update mean
        difference = np.array([[measurement[2] - range_expected],
                               [measurement[3] - bearing_expected]])
        
        innovation = K.dot(difference)
        particle.lm_mean[landmark_idx] += innovation.T[0]

        # Update covariance
        particle.lm_cov[landmark_idx] =\
            (np.identity(2) - K.dot(H_m)).dot(particle.lm_cov[landmark_idx])

        # Importance factor
        particle.weight = np.linalg.det(2 * np.pi * Q) ** (-0.5) *\
            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                    dot(difference))[0, 0]

        # Update timestamp
        particle.timestamp = measurement[0]


if __name__ == '__main__':
    pass
