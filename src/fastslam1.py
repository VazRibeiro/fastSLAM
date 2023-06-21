#!/usr/bin/env python3
'''
FastSlam 1 with Known Correspondences implementation.
See page 450 from Probablistic Robotics by Sebastian Thrun (Author),
Wolfram Burgard (Author) and Dieter Fox (Author).
'''

import math
import numpy as np
import csv
import os
import tf.transformations as tf
from scipy.spatial.transform import Rotation
import copy
import time
from particle import Particle
from motion_model import MotionModel
from measurement_model import MeasurementModel


class FastSLAM1():

    def __init__(self):
        '''
        Initialize models, particles and array of predicted positions
        '''
        self.depth = 1      # 3 meters of depth
        self.slope = 1.5   # 30 degree aperture
        # Initialize Motion Model object
        # [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6]
        motion_noise = np.array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2])*0.7
        self.motion_model = MotionModel(motion_noise)
        # Initialize Measurement Model object
        Q = np.array([[0.05, 0.0],[0.0, 0.05]])
        self.measurement_model = MeasurementModel(Q)
        # Initial Pose [ x, y, zAxis_rotation]
        initial_pose  = [0,0,0]
        self.odom_x = initial_pose[0]
        self.odom_y = initial_pose[1]
        self.odom_theta = initial_pose[2]
        # Array of N particles
        self.N_particles = 100
        # Initial position
        initial_variance = np.array([0,0,0])
        # Create particles
        self.particles = []
        self.timestamp = 0
        for i in range(self.N_particles):
            # Initialize the particle
            particle=Particle(initial_pose,self.N_particles)
            # Apply Gaussian noise to initial position of the particles
            particle.x = np.random.normal(particle.x, initial_variance[0])
            particle.y = np.random.normal(particle.y, initial_variance[1])
            particle.theta = np.random.normal(particle.theta, initial_variance[2])
            self.particles.append(particle)
        # Initialize the array to keep the average position
        self.predicted_position = np.array([[0,0,0]])
        self.odometry = np.array([[0,0,0]])
        self.measured_ids = np.zeros((0,1))
        self.accepted_landmarks = [33, 32]
        self.iterations = 0


    def odometry_update(self, control):
        '''
        Update the estimation of the robot position using the previous position and
        the control action [timestamp v, w]. Also update odometry position.
        '''
        # Initialize time in particles
        if self.timestamp == 0:
            self.timestamp = control[0]
            return
        # Update odometry estimate
        x_t = self.motion_model.sample_real_model_velocity(
            self.timestamp, 
            self.odom_x, 
            self.odom_y, 
            self.odom_theta, 
            control)
        self.odom_x = x_t[0]
        self.odom_y = x_t[1]
        self.odom_theta = x_t[2]
        for particle in self.particles:
            # Update robot state estimate
            x_t = self.motion_model.sample_motion_model_velocity(
                self.timestamp, 
                particle.x, 
                particle.y, 
                particle.theta, 
                control)
            particle.x = x_t[0]
            particle.y = x_t[1]
            particle.theta = x_t[2]
            # Limit θ within [-pi, pi]
            if (particle.theta > np.pi):
                particle.theta -= 2 * np.pi
            elif (particle.theta < -np.pi):
                particle.theta += 2 * np.pi
        self.timestamp = control[0]


    def landmarks_update_known(self,measurements,resampler):
        '''
        Update landmark mean and covariance for all landmarks of all particles.
        Fastslam1.0 known correspondences
        '''
        # Check if measurements are empty
        if len(measurements.transforms) == 0:
            return
        # Check if measurements late
        if measurements.header.stamp.to_sec() < self.timestamp:
            #print("lost a message...")
            return
        # Loop all the measurements in the fiducial transform array
        for transform in measurements.transforms:
            x = transform.transform.translation.z
            y = transform.transform.translation.x
            bearing = np.arctan2(-y,x)
            depth = np.sqrt(x**2 + y**2)
            filtered_measurement = np.array([depth,bearing,x,-y])
            # Check if it's a new landmark
            if ~np.any(np.isin(self.measured_ids,transform.fiducial_id)):
            # if  (transform.fiducial_id not in self.measured_ids) and\
            #     (transform.fiducial_id in self.accepted_landmarks):
                # If it's a new landmark, add it
                self.measured_ids = np.append(self.measured_ids,[[transform.fiducial_id]],0)
                for particle in self.particles:
                    index = len(self.measured_ids)-1
                    # Initialize mean and covariance for this landmark
                    self.measurement_model.initialize_landmark(
                        particle,
                        filtered_measurement,
                        index
                        )
            else:
            # elif transform.fiducial_id in self.accepted_landmarks:
                for particle in self.particles:
                    index = np.where(self.measured_ids == transform.fiducial_id)[0][0]
                    self.measurement_model.landmark_update_known(
                        particle,
                        filtered_measurement,
                        index
                        )
        # Normalize all weights
        self.weights_normalization()
        # Resample all particles according to the weights
        self.importance_sampling(resampler)


    def landmarks_update_unknown(self,measurements,resampler):
        '''
        Update landmark mean and covariance for all landmarks of all particles.
        Fastslam1.0 unknown correspondences
        '''
        # Check if measurements are empty
        if len(measurements.transforms) == 0:
            return
        # Check if measurements late
        if measurements.header.stamp.to_sec() < self.timestamp:
            #print("lost a message...")
            return
        # For each particle update the landmarks based on the measurement
        for particle in self.particles:
            particle.was_seen[:] = 0
            particle.should_been_seen[:] = 0
            # for each new measurement 
            for transform in measurements.transforms:
                x = transform.transform.translation.z
                y = -transform.transform.translation.x
                bearing = np.arctan2(y,x)
                depth = np.sqrt(x**2 + y**2)
                filtered_measurement = np.array([depth,bearing,x,y])
                # If it's the first landmark
                if len(particle.measured_ids) == 0:
                    # Initialize the particle confidence threshold
                    particle.confidence = np.append(particle.confidence,[[1]],0)
                    particle.measured_ids = np.append(particle.measured_ids,[[0]],0)
                    particle.was_seen = np.append(particle.was_seen,[[1]],0)
                    particle.should_been_seen = np.append(particle.should_been_seen,[[1]],0)
                    # Initialize mean and covariance for this landmark
                    self.measurement_model.initialize_landmark(
                        particle,
                        filtered_measurement,
                        0
                        )
                # If it's not the first landmark
                else:
                    # for this particle, compute a match for this measurement
                    weight = np.zeros((len(particle.measured_ids)+1,1))
                    for id in range(len(particle.measured_ids)):
                        weight[id] = self.measurement_model.match_landmark(
                            particle,
                            filtered_measurement,
                            id
                            )
                    weight[len(particle.measured_ids)] = 1/self.N_particles
                    most_likely_id = np.argmax(weight)
                    number_landmarks = max(len(particle.measured_ids),most_likely_id+1)
                    # Now that the match is found update the landmark data for the
                    # new observation.
                    for id in range(number_landmarks):
                        # If it's a new landmark:
                        if id == most_likely_id and most_likely_id ==len(particle.measured_ids):
                            # Initialize the particle confidence threshold
                            particle.confidence = np.append(particle.confidence,[[1]],0)
                            particle.was_seen = np.append(particle.was_seen,[[1]],0)
                            particle.should_been_seen = np.append(particle.should_been_seen,[[1]],0)
                            particle.measured_ids = np.append(
                                particle.measured_ids,
                                # avoid repeating ids when ids are removed
                                [[np.max(particle.measured_ids)+1]],
                                0
                                )
                            # Initialize mean and covariance for this landmark
                            self.measurement_model.initialize_landmark(
                                particle,
                                filtered_measurement,
                                most_likely_id
                                )
                        # if it's an observed landmark
                        elif id == most_likely_id and most_likely_id<len(particle.measured_ids):
                            particle.was_seen[id] = 1
                            self.measurement_model.landmark_update_known(
                                particle,
                                filtered_measurement,
                                most_likely_id
                                )
                            # Increase particle confidence everytime it's seen
                            particle.confidence[most_likely_id] += 1
                        else:
                            # Compute the relative position vector
                            dx = particle.mean[id,0] - particle.x
                            dy = particle.mean[id,1] - particle.y        
                            # Apply rotation to align the vector with the robot's reference frame
                            x_local = dx * math.cos(particle.theta) + dy * math.sin(particle.theta)
                            y_local = -dx * math.sin(particle.theta) + dy * math.cos(particle.theta)
                            # If the landmark is not within robot fov, skips landmark
                            if self.check_fov(x_local, y_local, self.depth,self.slope) == False:
                                continue
                            # If the landmark was in the FoV set the flag
                            particle.should_been_seen[id] = 1
            # Now that all measurements were processed, update the confidence
            # of each landmark in each particle
            ids_to_remove = []
            for id in range(len(particle.measured_ids)):
                if particle.was_seen[id] == 0 and particle.should_been_seen[id] == 1:
                    # reduce confidence in the particles that should've been seen, but were not
                    particle.confidence[id] -= 1
                    # If the confidence drops too much, save the id to be removed
                    if particle.confidence[id] < 0:
                        ids_to_remove.append(id)
            # Remove all the bad ids at once
            particle.mean = np.delete(particle.mean, ids_to_remove, axis=0)
            particle.cov = np.delete(particle.cov, ids_to_remove, axis=0)
            particle.confidence = np.delete(particle.confidence, ids_to_remove, axis=0)
            particle.measured_ids = np.delete(particle.measured_ids, ids_to_remove, axis=0)
            particle.should_been_seen = np.delete(particle.should_been_seen, ids_to_remove, axis=0)
            particle.was_seen = np.delete(particle.was_seen, ids_to_remove, axis=0)
        # Normalize all weights
        self.weights_normalization()
        # Resample all particles according to the weights
        self.importance_sampling(resampler)


    def weights_normalization(self):
        '''
        Normalize weight in all particles so that the sum = 1.
        '''
        # Compute sum of the weights
        sum = 0.0
        for particle in self.particles:
            sum += particle.weight
        # If sum is too small, equally assign weights to all particles
        if sum < 1e-10:
            for particle in self.particles:
                particle.weight = 1.0 / self.N_particles
            return
        for particle in self.particles:
            particle.weight /= sum


    def importance_sampling(self,resampler):
        '''
        Resampling using 2 possible methods: always resample or selective
        re-sampling.
        '''
        # Construct weights vector
        weights = []
        sum = 0.0
        # method 1
        if resampler == 'simple':
            for particle in self.particles:
                weights.append(particle.weight)
            # Resample all particles according to importance weights
            new_indexes = np.random.choice(
                len(self.particles), 
                len(self.particles),
                replace=True, 
                p=weights
                )
            # Update new particles
            new_particles = []
            for index in new_indexes:
                new_particles.append(copy.deepcopy(self.particles[index]))
            self.particles = new_particles
        # method 2
        elif resampler == 'selective':
            for particle in self.particles:
                weights.append(particle.weight)
                sum += (particle.weight)**2
            n_eff = 1/sum
            if n_eff<(self.N_particles/2):
                # Resample all particles according to importance weights
                new_indexes = np.random.choice(
                    len(self.particles), 
                    len(self.particles),
                    replace=True, 
                    p=weights
                    )
                # Update new particles
                new_particles = []
                for index in new_indexes:
                    new_particles.append(copy.deepcopy(self.particles[index]))
                self.particles = new_particles
        


    def save_data(self):
        '''
        Calculate the average position of the particles
        '''
        x = 0.0
        y = 0.0
        theta = 0.0
        for particle in self.particles:
            x += particle.x
            y += particle.y
            theta += particle.theta
        x /= self.N_particles
        y /= self.N_particles
        theta /= self.N_particles
        # If the position changed enough, save the new estimate
        if np.linalg.norm(self.predicted_position[-1,0:1] - [x,y]) > 0.1\
        or np.sqrt((self.predicted_position[-1,2] - theta)**2) > 0.08:
            self.predicted_position = np.append(
                self.predicted_position, 
                [[x,y,theta]], 
                axis=0
                )
            self.odometry = np.append(
                self.odometry, 
                [[self.odom_x,self.odom_y,self.odom_theta]], 
                axis=0
                )
    

    def write_data_to_files(self,x_truth,y_truth,method):
        # Extract the weights from the particles to find the best one
        weights = np.array([particle.weight for particle in self.particles])
        max_index = np.argmax(weights)
        # Get the ids of the particle with highest weight
        if method == 'known':
            ids = self.measured_ids
        elif method == 'unknown':
            ids = self.particles[max_index].measured_ids
        # Get the mean of the particle with highest weight
        mean = self.particles[max_index].mean
        mean_values = []
        for index in range(len(ids)):
            mean_values.extend([mean[index,0], mean[index,1],ids[index,0]])

        # Get the correct directory to the simulate_data_for_fastslam
        # package, where the data will be created in the output directory.
        current_dir = os.path.abspath(__file__)
        current_dir = current_dir.rstrip('fastslam1.py')
        current_dir = current_dir.rstrip('/src')
        current_dir = current_dir.rstrip('/fastSLAM')
        file_path1 = "simulate_data_for_fastslam/output/points_predict.txt"
        file_path2 = "simulate_data_for_fastslam/output/landmarks_predict.txt"
        # Specify the file path
        file_path1 = os.path.join(current_dir, file_path1)
        file_path2 = os.path.join(current_dir, file_path2)
        # Write to the positions file
        with open(file_path1, 'a', newline='') as file:  # Open the file in write mode
            writer = csv.writer(file, delimiter=',')
            # Write the data to the file
            row = [self.predicted_position[-1, 0], self.predicted_position[-1, 1],
                x_truth, y_truth, self.iterations, self.odometry[-1, 0],
                self.odometry[-1, 1]]
            writer.writerow(row)
            self.iterations += 1
        # Write to the landmarks file
        with open(file_path2, 'a', newline='') as file:  # Open the file in write mode
            writer = csv.writer(file, delimiter=',')
            # Write the data to the file
            row = [self.iterations] + mean_values
            writer.writerow(row)
            self.iterations += 1


    def get_plot_data(self,method):
        '''
        Get data to pass to the plotting process
        '''
        # Extract the weights from the particles to find the best one
        weights = np.array([particle.weight for particle in self.particles])
        max_index = np.argmax(weights)
        # Get the data for particles position
        x = [particle.x for particle in self.particles]
        y = [particle.y for particle in self.particles]
        # Get the ids of the particle with highest weight
        if method == 'known':
            ids = self.measured_ids
        elif method == 'unknown':
            ids = self.particles[max_index].measured_ids
        # Get the mean and covariance of the particle with highest weight
        mean = self.particles[max_index].mean
        cov = self.particles[max_index].cov
        return self.predicted_position, x, y, ids, mean, cov, self.odometry


    def check_fov(self,x_fid,y_fid,depth,slope):
        # Current fov at 45º angle
        if abs(y_fid)*slope <= x_fid and 0 < x_fid < depth:
            return True
        else:
            return False


if __name__ == "__main__":
    pass
