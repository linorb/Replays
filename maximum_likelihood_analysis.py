"""Replay analysis - ground play
"""
import matlab
import activity_loading
import maximum_likelihood

import numpy as np
import matplotlib.pyplot as plt

events_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\Replay_Pilot\C32M2\EnvA_am\finalResults\finalEventsMat.mat'
frame_log_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\Replay_Pilot\C32M2\EnvA_am\finalResults\frameLog.csv'
movment_filename = r'Z:\Short term data storage\Lab members\Yoav\All pre-processed data\Replay_Pilot\movment_data\C32M2_Day1.mat'

events_mat = matlab.load_events_file(events_filename)
frames_indices = matlab.load_frame_log_file(frame_log_filename)
movment_data = matlab.load_mvmt_file(movment_filename)

events_trace = activity_loading.order_events_into_trials(events_mat, frames_indices)

# Train maximum likelihood decoder on all trials but one and test on the remaining
# trial
linear_trials_indices =[3,4,5,6]
[train_bins, train_events] = activity_loading.create_training_data(movment_data, events_trace, linear_trials_indices)
p_neuron_bin = maximum_likelihood.calculate_p_r_s_matrix(train_bins, train_events)

[test_bins, test_events] = activity_loading.create_training_data(movment_data, events_trace, 7)

estimated_bins = maximum_likelihood.decode_entire_trial (test_events, p_neuron_bin)

plt.figure(1)
plt.plot(test_bins,'r')
plt.plot(estimated_bins,'b')
plt.legend('Real bins','Estimated bins')