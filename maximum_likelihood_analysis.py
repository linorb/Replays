"""Replay analysis - playground
"""
import bambi.tools.matlab
import bambi.tools.activity_loading
import bambi.analysis.maximum_likelihood

import matplotlib.pyplot as plt

events_filename = r'D:\data_for_analyzing_real_time_event_detector\c40m3_day1\events.mat'
frame_log_filename = r'D:\data_for_analyzing_real_time_event_detector\c40m3_day1\frameLog.csv'
movment_filename = r'D:\data_for_analyzing_real_time_event_detector\c40m3_day1\behavioral.mat'

events_mat = bambi.tools.matlab.load_events_file(events_filename)
frames_indices = bambi.tools.matlab.load_frame_log_file(frame_log_filename)
movment_data = bambi.tools.matlab.load_mvmt_file(movment_filename)

events_trace = bambi.tools.activity_loading.order_events_into_trials(events_mat, frames_indices)

# Train maximum likelihood decoder on all trials but one and test on the remaining
# trial
linear_trials_indices =[2,3,5,6]
[train_bins1, train_events] = bambi.tools.activity_loading.create_training_data(movment_data, events_trace, linear_trials_indices)

train_bins = bambi.tools.activity_loading.wide_binning(train_bins1,24, 3)
p_neuron_bin = bambi.analysis.maximum_likelihood.create_delayed_p_r_s(train_bins, train_events, 20)

[test_bins1, test_events] = bambi.tools.activity_loading.create_training_data(movment_data, events_trace, [4])
test_bins = bambi.tools.activity_loading.wide_binning(test_bins1,24, 3)

estimated_bins, estimated_prob = bambi.tools.activity_loading.decode_entire_trial (test_events, p_neuron_bin, 20)

fig, axx = plt.subplots(1,2)
axx[0].plot(test_bins,'r')
axx[0].plot(estimated_bins,'b')
axx[1].plot(estimated_prob)
fig.show()
raw_input('press enter')
