# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 10:22:51 2016

@author: Administrator
"""
import sys
sys.path.append('C:\Anaconda2\Lib\site-packages\odo\backends')
sys.path.append('D:\\dev\\real_time_imaging')
sys.path.append('D:\\dev\\replays')

import tools.matlab
import tools.activity_loading
import tools.maximum_likelihood
import tools.dilute

import numpy as np
import matplotlib.pyplot as plt

events_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\Replay_Pilot\C32M2\EnvA_am\finalResults\finalEventsMat.mat'
frame_log_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\Replay_Pilot\C32M2\EnvA_am\finalResults\frameLog.csv'
movment_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\Replay_Pilot\movment_data\C32M2_Day1.mat'

events_mat = tools.matlab.load_events_file(events_filename)
frames_indices = tools.matlab.load_frame_log_file(frame_log_filename)
movment_data = tools.matlab.load_mvmt_file(movment_filename)
cell_registration = tools.matlab.load_cell_registration\
                         (cell_registration_filename)

events_trace = tools.activity_loading.order_events_into_trials\
                    (events_mat, frames_indices)

# Train maximum likelihood decoder on all trials but one and test on the 
# remaining trial
linear_trials_indices =[2,3,4,5]

[train_bins, train_events] = tools.activity_loading.create_training_data\
                                  (movment_data, events_trace,\
                                   linear_trials_indices)
                                    
train_velocity = tools.activity_loading.concatenate_movment_data\
                      (movment_data, 'velocity', linear_trials_indices)

# For Yoav specific data in 30 Hz there is need to dilute the bins since there
# was a mistake in recording exposore time

dilute_bins = tools.dilute.dilute_array_by_2\
                   (train_bins, np.int_(train_events.shape[1]))
                
train_bins = dilute_bins.astype(int)

dilute_velocity = tools.dilute.dilute_array_by_2\
                       (train_velocity, np.int_(train_events.shape[1]))
                    
train_velocity = dilute_velocity.astype(int)

# choose only frames of running (without edges, and with velocity threshold)
running_indices = np.logical_and(train_bins > 22, train_bins > 1)
running_indices = np.logical_and(running_indices, np.absolute(train_velocity)>1)
train_bins = train_bins[running_indices]
train_events = train_events[:, running_indices]

# create p_neuron_bin for the training data
p_neuron_bin = tools.maximum_likelihood.calculate_p_r_s_matrix\
                    (train_bins, train_events)

# choose test data
[test_bins, test_events] = tools.activity_loading.create_training_data\
                                (movment_data, events_trace, [6])
                                
test_velocity = tools.activity_loading.concatenate_movment_data\
                     (movment_data, 'velocity', [6])

# For Yoav specific data in 30 Hz there is need to dilute the bins since there
# was a mistake in recording exposore time

dilute_bins = tools.dilute.dilute_array_by_2\
                    (test_bins, np.int_(test_events.shape[1]))
                    
test_bins = dilute_bins.astype(int)
dilute_velocity = tools.dilute.dilute_array_by_2\
                       (test_velocity, np.int_(test_events.shape[1]))
                       
test_velocity = dilute_velocity.astype(int)

# decode:
estimated_bins = tools.activity_loading.decode_entire_trial\
                      (test_events, p_neuron_bin)

plt.figure(1)
plt.plot(test_bins,'r')
plt.plot(estimated_bins,'b')
plt.ylim(-1, 24)
plt.legend('Real bins','Estimated bins')