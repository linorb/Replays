import numpy as np
import os
from matplotlib.pyplot import *
import scipy.stats

from bambi.tools import matlab
from bambi.tools.activity_loading import *
from bambi.analysis.maximum_likelihood import *
from zivlab.analysis.place_cells import find_place_cells

EDGE_BINS = [0, 1, 10, 11]
VELOCITY_THRESHOLD = 5

# Linear track parameters
FRAME_RATE = [10]*5 #Hz
FRAME_RATE.extend([20]*4)
MOUSE = [3, 6, 6, 3, 0, 4, 4, 1, 1]
CAGE = [40, 40, 38, 38, 38, 6, 7, 11, 13]
ENV = [r'\linear']*5
ENV.extend([r'\envA']*4)
WORK_DIR = [r'D:\dev\replays\work_data\recall']*5
WORK_DIR.extend([r'D:\dev\replays\work_data\two_environments']*4)

# L shape parameters
# FRAME_RATE = [20]*4 #Hz
# MOUSE = [4, 4, 1, 1]
# CAGE = [6, 7, 11, 13]
# ENV= [r'\envB']*4
# WORK_DIR = [r'D:\dev\replays\work_data\two_environments']*4


def load_session_data(session_dir):
    # Load events, traces, and behavioral data (my_mvmt) for entire session
    events_filename = 'finalEventsMat.mat'
    traces_filename = 'finalTracesMat.mat'
    log_filename = 'frameLog.csv'
    behavior_filename = 'my_mvmt_smooth.mat'

    all_events = matlab.load_events_file(os.path.join(session_dir,
                                                      events_filename))
    all_traces = matlab.load_traces_matrix(os.path.join(session_dir,
                                                        traces_filename))
    frame_log = matlab.load_frame_log_file(os.path.join(session_dir,
                                                        log_filename))
    movement_data = matlab.load_mvmt_file(os.path.join(session_dir,
                                                       behavior_filename))

    events_divided_to_trials = order_events_into_trials(all_events, frame_log)
    linear_trials_indices = range(len(events_divided_to_trials))[1:-1]
    [bins, events] = create_training_data(movement_data,
                                          events_divided_to_trials,
                                          linear_trials_indices)
    bins = wide_binning(bins, 24, 2)
    velocity = concatenate_movment_data(movement_data, 'velocity',
                                        linear_trials_indices)
    forward_velocity = velocity > VELOCITY_THRESHOLD
    backward_velocity = velocity < -VELOCITY_THRESHOLD
    place_cells_forward, _, _ = find_place_cells(bins[forward_velocity],
                                                 events[:, forward_velocity],
                                                 min_number_of_events=15)
    place_cells_backward, _, _ = find_place_cells(bins[backward_velocity],
                                                  events[:, backward_velocity],
                                                  min_number_of_events=15)
    place_cells = np.unique(np.concatenate([place_cells_forward,
                                            place_cells_backward]))
    velocity_ind = np.abs(velocity) > VELOCITY_THRESHOLD
    p_r_s = calculate_p_r_s_matrix(bins[velocity_ind],
                                  events[place_cells, :]\
                                  [:,velocity_ind])


    events = order_events_into_trials(all_events, frame_log)
    traces = order_events_into_trials(all_traces, frame_log)

    return events, traces, movement_data, place_cells, p_r_s

def calculate_conditional_activity_probability(events_segments):
    # Calculate conditional probability for events in segments of activity.
    # Args:
    # segments_activity: A list of segments, where each segment is a list of two
    #                    segments in time of neuronal activity. for example,
    #                    [activity vectors from two seconds before run, activity vectors
    #                     from the run epoch]. the conditional probability calculated in
    #                    this example will be:
    #                    p(active before run|active during run) for each neuron
    run_activity = []
    edge_activity = []
    for segment in events_segments:
        edge_activity.append(np.sum(segment[0], axis=1) > 0)
        run_activity.append(np.sum(segment[1], axis=1) > 0)

    run_activity = np.vstack(run_activity).T
    edge_activity = np.vstack(edge_activity).T

    # f, axx = subplots(1, 3, sharey=True, sharex=True)
    # axx[0].imshow(run_activity, interpolation='none', aspect='auto')
    # axx[0].set_title('run activity')
    # axx[1].imshow(edge_activity, interpolation='none', aspect='auto')
    # axx[1].set_title('edge activity')
    # axx[2].imshow(edge_activity*run_activity, interpolation='none', aspect='auto')
    # axx[2].set_title('multiplication')
    # f.show()

    p_run = np.sum(run_activity, axis=1)/np.float32(run_activity.shape[1])

    # Calculating the conditional probability
    edge_run_activity = run_activity*edge_activity
    edge_non_run_activity = ~run_activity*edge_activity
    number_of_active_runs = np.sum(run_activity, axis=1)
    number_of_non_active_runs = np.sum(~run_activity, axis=1)
    p_edge_run = np.sum(edge_run_activity, axis=1)/np.float32(number_of_active_runs)
    p_edge_non_run = np.sum(edge_non_run_activity, axis=1)/np.float32(number_of_non_active_runs)
    p_edge = np.sum(edge_activity, axis=1)/np.float32(edge_activity.shape[1])

    # f1, axx = subplots(2, 1, sharey=True, sharex=True)
    # hist_bins = np.arange(0, 1.05, 0.05)
    # axx[0].hist(p_edge[~np.isnan(p_edge)], bins = hist_bins)
    # axx[0].set_title('p(active in edge)')
    # axx[1].hist(p_edge_run[~np.isnan(p_edge_run)], bins = hist_bins)
    # axx[1].set_title('p(active in edge| active in run)')
    # f1.show()
    # raw_input('press enter')

    return p_edge_run, p_edge_non_run, p_edge

def create_segments_for_run_epochs_and_edges_entire_session(activity,
                                                            movement_data,
                                                            segment_type,
                                                            seconds_range,
                                                            edge_bins,
                                                            frame_rate):
    # Create segments for using in calculate_conditional_activity_probability
    #  afterwards that divide the session to run epochs and the activity that
    #  is done in the edges before/after the run.
    # Args:
    #     activity: Either events/traces matrix, as created from load_session_data movement_data:
    #     As created from load_session_data
    #
    #     segment_type: A string of 'before' or 'after'
    #
    #     seconds_range: A list of the range of seconds to take. for
    #     example to take the two seconds before run: segment_type='before',
    #     seconds_range=[0 2]. to take the two seconds that starts 8 seconds
    #     after run epoch ends: segment_type='after', seconds_range=[8 10]
    #
    #     edge_bins: A list of the edge bins
    session_segments = []
    number_of_trials = len(activity)
    for i in range(1,number_of_trials-1):
        trial_events = activity[i]
        bins = movement_data[i]['bin']
        bins = wide_binning(bins, 24, 2)
        trial_segments = create_segments_for_run_epochs_and_edges_for_trial\
                                                    (trial_events, bins,
                                                    segment_type, seconds_range,
                                                    edge_bins, frame_rate)
        session_segments.extend(trial_segments)

    return session_segments



def create_segments_for_run_epochs_and_edges_for_trial(events, bins,
                                                       segment_type,
                                                       seconds_range,
                                                       edge_bins,
                                                       frame_rate):
    # Create segments for one trial.
    # Args:
    #   events: A matrix of events\traces. size [#neurons, #frames]
    #   bins: An array of bins. size [#frames]
    #   segment_type: A string of 'before' or 'after'
    #   seconds_range: A list of the range of seconds to take. for example
    #                to take the two seconds before run: segment_type='before',
    #                seconds_range=[0 2]. to take the two seconds that starts
    #                8 seconds after run epoch ends: segment_type='after',
    #                seconds_range=[8 10]
    #   edge_bins: A list of the edge bins
    edge_bins_mask = np.zeros_like(bins, dtype=bool)
    for b in edge_bins:
        edge_bins_mask[bins == b] = True

    edge_masked = np.ma.array(bins, mask=edge_bins_mask)
    run_locations = np.ma.flatnotmasked_contiguous(edge_masked)

    run_masked = np.ma.array(bins, mask=~edge_bins_mask)
    edge_locations = np.ma.flatnotmasked_contiguous(run_masked)

    if segment_type == 'before':
        # Make sure that the first segment in this case is edge
        if run_locations[0].start < edge_locations[0].start:
            run_locations = run_locations[1:]
    elif segment_type == 'after':
        # Make sure that the first segment in this case is run epoch
        if run_locations[0].start > edge_locations[0].start:
            edge_locations = edge_locations[1:]

    number_of_segments = min(len(edge_locations), len(run_locations))
    segments_activity = []
    for i in range(number_of_segments):
        segment = []
        edge_segment = events[:, edge_locations[i]]
        if seconds_range:
            frames_indices = np.array(seconds_range) * frame_rate
            try:
                if segment_type == 'before':
                    if frames_indices[0]>0:
                        edge_segment = edge_segment[:, -frames_indices[1]:
                                                    -frames_indices[0]]
                        bins_edge_segment = bins[edge_locations[i]]\
                                        [-frames_indices[1]:-frames_indices[0]]
                    else:
                        edge_segment = edge_segment[:, -frames_indices[1]:]
                        bins_edge_segment = bins[edge_locations[i]]\
                                                [-frames_indices[1]:]
                elif segment_type == 'after':
                    edge_segment = edge_segment[:, frames_indices[0]:
                                                    -frames_indices[1]]
                    bins_edge_segment = bins[edge_locations[i]]\
                                        [frames_indices[0]:-frames_indices[1]]
            except IndexError:
                continue

        segment.append(edge_segment)
        run_segment = events[:, run_locations[i]]
        segment.append(run_segment)

        #plotting for debugging
        # f, axx = subplots(2, 2, sharex='col')
        # f.tight_layout()
        # f.subplots_adjust(top=0.9)
        # axx[0, 0].plot(bins_edge_segment)
        # axx[0, 1].plot(bins[run_locations[i]])
        # axx[1, 0].imshow(edge_segment, interpolation='none', aspect='auto')
        # axx[1, 1].matshow(run_segment, interpolation='none', aspect='auto')
        # f.show()
        # raw_input('press enter to continue')
        # close(f)

        segments_activity.append(segment)

    return segments_activity

def t_test_for_deppendent_smaples(a, b):
    # Calculate t test for difference. omit "nan"s
    nan_inds = np.isnan(b) | np.isnan(a)
    A = a[~nan_inds]
    B = b[~nan_inds]
    difference = A-B
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    tt = np.float32(mean_diff)/(std_diff/np.sqrt(len(A)))
    pval = scipy.stats.t.cdf(tt, len(A) - 1)

    return tt, 1-pval

def plot_segment_activity(events_segment, trace_segment):
    # Plot the traces and events of specific segment, i.e only the neurons
    # that were active in that segment (according
    # to events matrix. segment [0] - is the edge segment,
    # segment[1] - is the run segment
    total_neurons_activity = np.sum(events_segment[1], axis=1)
    neurons_mask = total_neurons_activity > 0
    relevent_event_segment = []
    relevent_trace_segment = []
    relevent_event_segment.append(events_segment[0][neurons_mask, :] > 0)
    relevent_event_segment.append(events_segment[1][neurons_mask, :] > 0)
    relevent_trace_segment.append(trace_segment[0][neurons_mask, :])
    relevent_trace_segment.append(trace_segment[1][neurons_mask, :])

    # calculate the mean std of the traces in order to plot them on the top of
    # each other with constant spacing
    mean_std = []
    mean_std.append(np.mean(np.std(trace_segment[0], axis=1)))
    mean_std.append(np.mean(np.std(trace_segment[1], axis=1)))

    f, axx = subplots(1, 2, sharey=True)
    number_of_traces = trace_segment[0].shape[0]
    number_of_frames_before = trace_segment[0].shape[1]
    number_of_frames_after = trace_segment[1].shape[1]
    for i in range(number_of_traces):
        # fix the traces values to fit
        traces_before = relevent_trace_segment[0][i, :] + i * mean_std[0]
        events_before = relevent_event_segment[0][i, :] * \
                        relevent_trace_segment[0][i, :]

        axx[0].plot(range(number_of_frames_before), traces_before)
        axx[0].plot(np.arange(number_of_frames_before)[events_before > 0],
                    events_before[events_before > 0], 'ro')

        axx[0].set_xlabel('#Frame')
        axx[0].set_title('Rest segment activity')

        traces_after = relevent_trace_segment[1][i, :] + i * mean_std[1]
        events_after = relevent_event_segment[1][i, :] * \
                       relevent_trace_segment[1][i, :]

        axx[1].plot(range(number_of_frames_after), traces_after)
        axx[1].plot(np.arange(number_of_frames_after)[events_after > 0],
                    events_after[events_after > 0], 'ro')

        axx[1].set_xlabel('#Frame')
        axx[1].set_title('Run segment activity')

    return

def plot_random_segment_activities(events_segments, trace_segments,
                                   number_of_segments_to_plot):
    number_of_segments = len(events_segments)
    indices_to_plot = np.random.permutation(number_of_segments)\
                        [:number_of_segments_to_plot]
    for i in indices_to_plot:
        print i
        plot_segment_activity(events_segments[i], trace_segments[i])
        raw_input('enter to continue')

    return

def normalize_trace_segment(trace_segment):
    number_of_traces = len(trace_segment[0])

    normalize_segment = []
    # Normalize each trace by its maximum
    normalize_segment[0] = trace_segment[0]/np.max(trace_segment[0],
                                                   axis=1)[:,None]
    normalize_segment[1] = trace_segment[1]/np.max(trace_segment[1],
                                                   axis=1)[:,None]

    return normalize_segment

def main():
    p_value = {}
    t_value = {}
    cohen_d = {}

    for i, mouse in enumerate(MOUSE):

        mouse_name = 'c%dm%d' %(CAGE[i], mouse)
        p_value[mouse_name] = {'p_before': [],
                               'p_after': [],
                               'p_before_after': []}
        t_value[mouse_name] = {'t_before': [],
                               't_after': [],
                               't_before_after': []}
        cohen_d[mouse_name] = {'d_before': [],
                               'd_after': [],
                               'd_before_after': []}
        mouse_dir = WORK_DIR[i] + '\c%dm%d' %(CAGE[i], mouse)
        days_list = [x[1] for x in os.walk(mouse_dir)][0]
        for day in days_list:
            print CAGE[i], mouse, day
            print
            session_dir = mouse_dir + '\%s\%s' %(day, ENV[i])
            events, traces, movement_data, _, _ = load_session_data(session_dir)
            events_segments_before = \
                create_segments_for_run_epochs_and_edges_entire_session(events,
                                                            movement_data,
                                                            'before', [0, 2],
                                                            EDGE_BINS,
                                                            FRAME_RATE[i])

            traces_segments_before = \
                create_segments_for_run_epochs_and_edges_entire_session(traces,
                                                            movement_data,
                                                            'before', [0, 2],
                                                            EDGE_BINS,
                                                            FRAME_RATE[i])
            # plot_random_segment_activities(events_segments_before,
            #                                traces_segments_before,
            #                                10)

            p_edge_run_before, p_edge_non_run_before, _ = \
                calculate_conditional_activity_probability(events_segments_before)

            stats, p = scipy.stats.ttest_rel(p_edge_run_before, p_edge_non_run_before,
                                             axis=0, nan_policy='omit')

            d = (np.nanmean(p_edge_run_before) - np.nanmean(p_edge_non_run_before)) /\
                (np.sqrt((np.nanstd(p_edge_run_before) ** 2 +
                          np.nanstd(p_edge_non_run_before) ** 2) / 2))

            p_value[mouse_name]['p_before'].extend([p])
            t_value[mouse_name]['t_before'].extend([stats])
            cohen_d[mouse_name]['d_before'].extend([d])

            events_segments_after = \
                create_segments_for_run_epochs_and_edges_entire_session(events,
                                                                movement_data,
                                                                'after', [2, 4],
                                                                EDGE_BINS,
                                                                FRAME_RATE[i])
            traces_segments_after = \
                create_segments_for_run_epochs_and_edges_entire_session(traces,
                                                                movement_data,
                                                                'after', [2, 4],
                                                                EDGE_BINS,
                                                                FRAME_RATE[i])
            # plot_random_segment_activities(events_segments_after,
            #                                traces_segments_after,
            #                                10)

            p_edge_run_after, p_edge_non_run_after, _ = \
                calculate_conditional_activity_probability(events_segments_after)

            stats, p = scipy.stats.ttest_rel(p_edge_run_after, p_edge_non_run_after,
                                             axis=0, nan_policy='omit')

            d = (np.nanmean(p_edge_run_after) - np.nanmean(p_edge_non_run_after)) / \
                (np.sqrt((np.nanstd(p_edge_run_after) ** 2 +
                          np.nanstd(p_edge_non_run_after) ** 2) / 2))

            p_value[mouse_name]['p_after'].extend([p])
            t_value[mouse_name]['t_after'].extend([stats])
            cohen_d[mouse_name]['d_after'].extend([d])

            stats, p = scipy.stats.ttest_rel(p_edge_run_before,
                                             p_edge_run_after, axis=0,
                                             nan_policy='omit')
            d = (np.nanmean(p_edge_run_before) - np.nanmean(
                p_edge_run_after)) / \
                (np.sqrt((np.nanstd(p_edge_run_after) ** 2 +
                          np.nanstd(p_edge_run_before) ** 2) / 2))

            p_value[mouse_name]['p_before_after'].extend([p])
            t_value[mouse_name]['t_before_after'].extend([stats])
            cohen_d[mouse_name]['d_before_after'].extend([d])

    np.savez('Linear_edge_statistics', p_value=p_value, t_value=t_value,
             cohen_d=cohen_d)

    raw_input('press enter to quit')

if __name__ == '__main__':
    main()