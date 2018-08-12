# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:13:07 2016

@author: Administrator
"""
import sys
sys.path.append('C:\Anaconda2\Lib\site-packages\odo\backends')
sys.path.append('D:\\dev\\real_time_imaging')
sys.path.append('D:\\dev\\replays')
sys.path.append('D:\\dev\\cell_registration')

import tools.matlab
import tools.activity_loading
import tools.maximum_likelihood
import tools.dilute

import numpy as np
import matplotlib.pyplot as plt

# Create a concatenated training data from several days:
events_files_path = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\NG\C33M4'
movement_files_path = r'Z:\Short term data storage\Lab members\Linor\C33M4'
cell_registration_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\NG\C33M4\13_7_16_Registered_C33M4\cellRegistered_Final_2 dimensional_0.5_14-Jul-2016_123700.mat'

cell_registration = tools.matlab.load_cell_registration\
                         (cell_registration_filename)

environment = 'EnvA'
ampm = ['am', 'pm', 'am', 'pm']
days = [1, 2, 3, 4]
events= []
bins = []
movment_data = []
events_trace = []
linear_trials_indices = np.arange(1, 5)

for i, day in enumerate(days):
    current_events_path = events_files_path + '\\Day' + str(day) + '\\' + \
                            environment + '_' + ampm[i] + '\\finalResults\\' 
                            
    current_movement_path = movement_files_path + '\\Day' + str(day) + '\\'
    
    events_filename = current_events_path + 'finalEventsMat.mat'
    
    frame_log_filename = current_events_path + 'frameLog.csv'
    
    movment_filename = current_movement_path + 'C33M4_Day' + str(day) + '_' + \
                        environment + '_' + ampm[i] 
    
    events_mat = tools.matlab.load_events_file(events_filename)
    frames_indices = tools.matlab.load_frame_log_file(frame_log_filename)
    movment_data.append(tools.matlab.load_mvmt_file(movment_filename))
    
    events_trace.append(tools.activity_loading.order_events_into_trials\
                        (events_mat, frames_indices))
    
    [train_bins, train_events] = tools.activity_loading.create_training_data \
                                       (movment_data[i], events_trace[i], \
                                       linear_trials_indices)
                                       
    number_of_frames = np.minimum(int(train_bins.shape[0]),\
                          int(train_events.shape[1]))
                              
    events.append(train_events[:number_of_frames])
    bins.append(train_bins[:number_of_frames])
    
all_events = tools.activity_loading.unite_sessions\
                  (events, [0, 3, 4, 7], cell_registration)

all_bins = bins[0]
for i in range(1,len(days)):
    all_bins = np.concatenate((all_bins, bins[i]))

all_bins = tools.activity_loading.wide_binning(all_bins, 24, 2)

# Calculate P(neuron|bin)    
p_neuron_bin = tools.maximum_likelihood.calculate_p_r_s_matrix\
                    (all_bins, all_events)

# Choose test data
[test_bins, test_events] = tools.activity_loading.create_training_data\
                                (movment_data[3], events_trace[3], [5])

test_bins = tools.activity_loading.wide_binning(test_bins, 24, 2)                        

test_events =  tools.activity_loading.unite_sessions\
                    ([test_events], [0], cell_registration)
                  
[estimated_bins, estimated_prob] = tools.activity_loading.decode_entire_trial\
                      (test_events, p_neuron_bin, 40)

plt.plot(test_bins,'r')
plt.plot(estimated_bins,'b')
plt.legend('Real bins','Estimated bins')  