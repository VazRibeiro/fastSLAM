#!/usr/bin/env python3
'''
Measurement model for Aruco Markers.
'''

import numpy as np


class MeasurementModel():
    def __init__(self, Q):
        '''
        Input: Q: Measurement noise 2x2 matrix.
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
        cov_matrix = H_inverse.dot(self.Q).dot(H_inverse.T)
        particle.cov = np.append(particle.cov,[cov_matrix],0)


    def compute_expected_measurement(self,particle,index):
        '''
        Compute the expected range and bearing given current robot state and
        landmark state.
        '''
        dx = particle.mean[index, 0] - particle.x
        dy = particle.mean[index, 1] - particle.y
        #print([index,dx,dy],particle.mean,particle.x,particle.y)
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


    def landmark_update_known(self,particle,measurement,index):
        '''
        Update landmark mean and covariance using EKF.
        This landmark has to be observed before.
        '''
        dx,dy,q =self.compute_expected_measurement(particle,index)
        range_exp = np.sqrt(q)
        if np.arctan2(dy, dx)*particle.theta<0 and \
           abs(np.arctan2(dy, dx) + particle.theta)+abs(particle.theta)>np.pi:
            bearing_exp = np.arctan2(dy, dx) + particle.theta
        else:
            bearing_exp = np.arctan2(dy, dx) - particle.theta

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle,dx,dy,q)
        # Compute measurement covariance
        Q = H_m.dot(particle.cov[index]).dot(H_m.T) + self.Q
        # Compute Kalman gain
        K = particle.cov[index].dot(H_m.T).dot(np.linalg.inv(Q))
        # Update mean
        difference = np.array([[measurement[0] - range_exp],
                               [measurement[1] - bearing_exp]])
        #print("id,meas, bear,arc,teta",id,measurement[1],bearing_exp,np.arctan2(dy, dx),particle.theta)
        innovation = K.dot(difference)
        particle.mean[index] += innovation.T[0]
        # Update covariance
        particle.cov[index]=(np.identity(2)-K.dot(H_m)).dot(particle.cov[index])
        # Importance factor
        particle.weight =   abs(np.linalg.det(2 * np.pi * Q)) ** (-0.5) *\
                            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                            dot(difference))[0,0]




    def match_landmark(self, particle, measurement, index):
        '''
        Compute the likelihood of a match between the measurement and the
        landmark number index+1.
        '''
        dx,dy,q =self.compute_expected_measurement(particle,index)
        range_exp = np.sqrt(q)
        if np.arctan2(dy, dx)*particle.theta<0 and \
           abs(np.arctan2(dy, dx) + particle.theta)+abs(particle.theta)>np.pi:
            bearing_exp = np.arctan2(dy, dx) + particle.theta
        else:
            bearing_exp = np.arctan2(dy, dx) - particle.theta

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle,dx,dy,q)
        # Compute measurement covariance
        Q = H_m.dot(particle.cov[index]).dot(H_m.T) + self.Q
        # Difference between measured and expected
        difference = np.array([[measurement[0] - range_exp],
                               [measurement[1] - bearing_exp]])
        # Importance factor
        weight =   abs(np.linalg.det(2 * np.pi * Q)) ** (-0.5) *\
                            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                            dot(difference))[0,0]

        return weight
    


if __name__ == '__main__':
    pass
