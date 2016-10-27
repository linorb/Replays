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
events_files_path = r'Z:\Long term data storage\Storwiz backup\Long term data storage\Data\2015\Two environments\2 days gap\C6_M4\Neuronal\Pre-processed Data\separated_sessions'
movement_files_path = r'Z:\Long term data storage\Storwiz backup\Long term data storage\Data\2015\Two environments\2 days gap\C6_M4\Behavioral'
cell_registration_filename = r'Z:\Long term data storage\Storwiz backup\Long term data storage\Data\2015\Two environments\2 days gap\C6_M4\Neuronal\Pre-processed Data\separated_sessions\Cell registration\cellRegistered_Final_2 dimensional_0.5_05-Sep-2016_113400.mat'

cell_registration = tools.matlab.load_cell_registration\
                         (cell_registration_filename)

environment = 'A'
ampm = ['am', 'am', 'pm', 'am']
days = [1, 3, 4, 5]
events= []
bins = []
movment_data = []
events_trace = []
linear_trials_indices = np.arange(1, 5)
session_indices = [0, 4, 7, 8]

for i, day in enumerate(days):
    current_events_path = events_files_path + '\\Session_' + str(day) + '_' + \
                            environment + '\\finalResults_ver2\\' 
                            
    current_movement_path = movement_files_path + '\\env' + environment + \
                            '\\Day_' + str(day) + '\\'
    
    events_filename = current_events_path + 'finalEventsMat.mat'
    
    frame_log_filename = current_events_path + 'frameLog.csv'
    
    movment_filename = current_movement_path + 'C6M4_Day' + str(day + 2) + \
                        '_' + environment + '_' + ampm[i] 
    
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
                  (events, session_indices, cell_registration)

all_bins = bins[0]
for i in range(1,len(days)):
    all_bins = np.concatenate((all_bins, bins[i]))

number_of_bins = 24
wide_bin_factor = 1
all_bins_wider = tools.activity_loading.wide_binning(all_bins, number_of_bins, \
                 wide_bin_factor)

# Calculate P(neuron|bin)    
p_neuron_bin = tools.maximum_likelihood.calculate_p_r_s_matrix\
                    (all_bins_wider, all_events)

# Choose test data
test_session_index = 1
test_trial_index = 5
[test_bins, test_events] = tools.activity_loading.create_training_data\
                                (movment_data[test_session_index], \
                                events_trace[test_session_index], \
                                [test_trial_index])

test_bins_wider = tools.activity_loading.wide_binning(test_bins, number_of_bins, \
                  wide_bin_factor)                        

test_events =  tools.activity_loading.unite_sessions\
                    ([test_events], [session_indices[test_session_index]], \
                    cell_registration)

number_of_frames_to_look_back = 35                
[estimated_bins, estimated_prob] = tools.activity_loading.decode_entire_trial\
                      (test_events, p_neuron_bin, \
                      number_of_frames_to_look_back)

plt.plot(test_bins_wider,'r')
plt.plot(estimated_bins,'b')
plt.legend('Real bins','Estimated bins')  