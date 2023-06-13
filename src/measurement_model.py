#!/usr/bin/env python3
'''
Measurement model for Aruco Markers.
'''

import numpy as np


class MeasurementModel():
    def __init__(self, Q):
        '''
        Input:
            Q: Measurement covariance matrix.
               Dimension: [2, 2].
        '''
        self.Q = Q


    def initialize_landmark(self,particle,measurement,index):
        '''
        Initialize landmark mean and covariance a landmark for given particle.
        Note: particles are already initialized with the default weight
        '''
        # Calculate position estimate of the landmark in the world referential
        x = particle.x + measurement[0] * np.cos(measurement[1] + particle.theta)
        y = particle.y + measurement[0] * np.sin(measurement[1] + particle.theta)
        # Initialize mean
        particle.mean = np.append(particle.mean,[[x,y]],0)
        # Expected measurement
        dx,dy,q =self.compute_expected_measurement(particle, index)
        # Initialize landmark Jacobian
        H_m = self.compute_landmark_jacobian(particle,dx,dy,q)
        # Initialize landmark covariance
        H_inverse = np.linalg.inv(H_m)
        particle.cov = H_inverse.dot(self.Q).dot(H_inverse.T)


    def compute_expected_measurement(self,particle,index):
        '''
        Compute the expected range and bearing given current robot state and
        landmark state.
        '''
        dx = particle.mean[index, 0] - particle.x
        dy = particle.mean[index, 1] - particle.y
        q = dx ** 2 + dy ** 2
        return dx,dy,q


    def compute_landmark_jacobian(self, particle,dx,dy,q):
        '''
        Computing the landmark Jacobian.
        Jacobian is given by the derivative: d h(x_t, x_l) / d (x_l)
        H_m =  dx/√q  dy/√q
               -dy/q  dx/q
        '''
        H_1 = np.array([dx/np.sqrt(q), dy/np.sqrt(q)])
        H_2 = np.array([-dy/q, dx/q])
        H_m = np.array([H_1, H_2])
        return H_m


    def landmark_update(self,particle,measurement,index):
        '''
        Update landmark mean and covariance using EKF.
        This landmark has to be observed before.
        '''
        # Compute expected measurement
        dx,dy,q =self.compute_expected_measurement(particle,index)
        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle,dx,dy,q)
        print('H_m')
        print(H_m)
        print('cov')
        print(particle.cov[index])
        
        # Compute Kalman gain
        Q = H_m.dot(particle.cov[index]).dot(H_m.T) + self.Q
        print('Q')
        print(Q)
        K = particle.cov[index].dot(H_m.T).dot(np.linalg.inv(Q))
        #K = particle.cov[index].dot(H_m.T).dot(Q)
        print('K gain')
        print(K)

        # Update mean
        difference = np.array([[measurement[2] - dx],
                               [measurement[3] - dy]])
        innovation = K.dot(difference)
        particle.mean[index] += innovation.T[0]
        # Update covariance
        particle.cov[index]=(np.identity(2)-K.dot(H_m)).dot(particle.cov[index])

        # Importance factor
        particle.weight =   np.linalg.det(2 * np.pi * Q) ** (-0.5) *\
                            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                            dot(difference))[0,0]


    def compute_correspondence(self, particle, measurement, index):
        '''
        Implementation for Fast SLAM 1.0.
        Compute the likelihood of correspondence for between a measurement and
        a given landmark.
        This process is the same as updating a landmark mean with EKF method.

        Input:
            particle: Particle() object to be updated.
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
            index: the index of the landmark (0 ~ 15).
        Output:
            likehood: likelihood of correspondence
        '''
        # Compute expected measurement
        range_expected, bearing_expected =\
            self.compute_expected_measurement(particle, index)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle, index)

        Q = H_m.dot(particle.cov[index]).dot(H_m.T) + self.Q
        difference = np.array([[measurement[2] - range_expected],
                               [measurement[3] - bearing_expected]])

        # likelihood of correspondence
        likelihood = np.linalg.det(2 * np.pi * Q) ** (-0.5) *\
            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                   dot(difference))[0, 0]

        return likelihood
    


if __name__ == '__main__':
    pass
