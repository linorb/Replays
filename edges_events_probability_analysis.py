import numpy as np
import os
from matplotlib.pyplot import *

from bambi.tools import matlab
from bambi.tools.activity_loading import order_events_into_trials

EDGE_BINS = [0, 1, 2, 21, 22, 23]
FRAME_RATE = 20 #Hz
MOUSE = [6, 4, 1, 1]
CAGE = [4, 7, 11, 13]
ENV = 'envA'
DAYS = '12345678'
WORK_DIR = r'D:\dev\replays\work_data\two_environments'

def load_session_data(session_dir):
    # Load events, traces, and behavioral data (my_mvmt) for entire session
    events_filename = 'finalEventsMat.mat'
    traces_filename = 'finalTracesMat.mat'
    log_filename = 'frameLog.csv'
    behavior_filename = 'my_mvmt.mat'

    all_events = matlab.load_events_file(os.path.join(session_dir, events_filename))
    all_traces = matlab.load_traces_matrix(os.path.join(session_dir, traces_filename))
    frame_log = matlab.load_frame_log_file(os.path.join(session_dir,log_filename))
    movement_data = matlab.load_mvmt_file(os.path.join(session_dir,behavior_filename))

    events = order_events_into_trials(all_events, frame_log)
    traces = order_events_into_trials(all_traces, frame_log)

    return events, traces, movement_data

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

    f, axx = subplots(1, 3, sharey=True, sharex=True)
    axx[0].imshow(run_activity, interpolation='none', aspect='auto')
    axx[0].set_title('run activity')
    axx[1].imshow(edge_activity, interpolation='none', aspect='auto')
    axx[1].set_title('edge activity')
    axx[2].imshow(edge_activity*run_activity, interpolation='none', aspect='auto')
    axx[2].set_title('multiplication')
    f.show()

    p_run = np.sum(run_activity, axis=1)/np.float32(run_activity.shape[1])

    # Calculating the conditional probability
    edge_run_activity = run_activity*edge_activity
    number_of_active_runs = np.sum(run_activity, axis=1)
    p_edge_run = np.sum(edge_run_activity, axis=1)/np.float32(number_of_active_runs)

    f1, axx = subplots(2, 1, sharey=True, sharex=True)
    hist_bins = np.arange(0, 1.01, 0.01)
    axx[0].hist(p_run[~np.isnan(p_run)], bins = hist_bins)
    axx[0].set_title('p(active in run)')
    axx[1].hist(p_edge_run[~np.isnan(p_edge_run)], bins = hist_bins)
    axx[1].set_title('p(active in edge| active in run)')
    f1.show()

    return p_edge_run, p_run

def create_segments_for_run_epochs_and_edges_entire_session(activity, movement_data,
                                                            segment_type, seconds_range,
                                                            edge_bins):
    # Create segments for using in calculate_conditional_activity_probability afterwards
    # that divide the session to run epochs and the activity that is done in the edges
    # before/after the run.
    # Args:
    #   activity: Either events/traces matrix, as created from load_session_data
    #   movement_data: As created from load_session_data
    #   segment_type: A string of 'before' or 'after'
    #   seconds_range: A list of the range of seconds to take. for example to take the
    #       two seconds before run: segment_type='before', seconds_range=[0 2]. to take the
    #       two seconds that starts 8 seconds after run epoch ends: segment_type='after',
    #       seconds_range=[8 10]
    #   edge_bins: A list of the edge bins
    session_segments = []
    number_of_trials = len(activity)
    for i in range(1,number_of_trials-1):
        trial_events = activity[i]
        bins = movement_data[i]['bin']
        trial_segments = create_segments_for_run_epochs_and_edges_for_trial(trial_events, bins,
                                                            segment_type, seconds_range,
                                                            edge_bins)
        session_segments.extend(trial_segments)

    return session_segments



def create_segments_for_run_epochs_and_edges_for_trial(events, bins,
                                                            segment_type, seconds_range,
                                                            edge_bins):
    # Create segments for one trial.
    # Args:
    #   events: A matrix of events\traces. size [#neurons, #frames]
    #   bins: An array of bins. size [#frames]
    #   segment_type: A string of 'before' or 'after'
    #   seconds_range: A list of the range of seconds to take. for example to take the
    #       two seconds before run: segment_type='before', seconds_range=[0 2]. to take the
    #       two seconds that starts 8 seconds after run epoch ends: segment_type='after',
    #       seconds_range=[8 10]
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
        frames_indices = np.array(seconds_range) * FRAME_RATE
        try:
            if segment_type == 'before':
                if frames_indices[0]>0:
                    edge_segment = edge_segment[:, -frames_indices[1]:-frames_indices[0]]
                else:
                    edge_segment = edge_segment[:, -frames_indices[1]:]
            elif segment_type == 'after':
                edge_segment = edge_segment[:, frames_indices[0]:-frames_indices[1]]
            segment.append(edge_segment)
        except IndexError:
            continue

        run_segment = events[:, run_locations[i]]
        segment.append(run_segment)

        segments_activity.append(segment)

    return segments_activity

def main():

    session_dir = r'D:\dev\replays\work_data\two_environments\c4m6\day1\envA'
    [events, traces, movement_data] = load_session_data(session_dir)
    activity_segments = create_segments_for_run_epochs_and_edges_entire_session(events, movement_data,
                                                            'before', [0,2],
                                                            EDGE_BINS)
    [p_edge_run, p_run] = calculate_conditional_activity_probability(activity_segments)

    raw_input('press enter to quit')

main()