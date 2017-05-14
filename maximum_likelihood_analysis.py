"""Replay analysis - playground
"""
import tools.matlab
import tools.activity_loading
import tools.maximum_likelihood

import matplotlib.pyplot as plt

events_filename = r'D:\data_for_analyzing_real_time_event_detector\c40m3_day1\events.mat'
frame_log_filename = r'D:\data_for_analyzing_real_time_event_detector\c40m3_day1\frameLog.csv'
movment_filename = r'D:\data_for_analyzing_real_time_event_detector\c40m3_day1\behavioral.mat'

events_mat = tools.matlab.load_events_file(events_filename)
frames_indices = tools.matlab.load_frame_log_file(frame_log_filename)
movment_data = tools.matlab.load_mvmt_file(movment_filename)

events_trace = tools.activity_loading.order_events_into_trials(events_mat, frames_indices)

# Train maximum likelihood decoder on all trials but one and test on the remaining
# trial
linear_trials_indices =[2,3,4,5]
[train_bins1, train_events] = tools.activity_loading.create_training_data(movment_data, events_trace, linear_trials_indices)

train_bins = tools.activity_loading.wide_binning(train_bins1,24, 3)
p_neuron_bin = tools.maximum_likelihood.calculate_p_r_s_matrix(train_bins, train_events)

[test_bins1, test_events] = tools.activity_loading.create_training_data(movment_data, events_trace, [6])
test_bins = tools.activity_loading.wide_binning(test_bins1,24, 3)

estimated_bins = tools.activity_loading.decode_entire_trial (test_events, p_neuron_bin)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(test_bins,'r')
ax.plot(estimated_bins[0],'b')
ax.legend('Real bins','Estimated bins')
