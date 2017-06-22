# this module finds and make statistical calculations on synchronous calcium
# events (SCEs)

import numpy as np
from matplotlib.pyplot import *
import matplotlib.gridspec as gridspec

from edges_events_probability_analysis import \
    create_segments_for_run_epochs_and_edges_entire_session,\
    load_session_data, EDGE_BINS, MOUSE, CAGE, ENV, DAYS, WORK_DIR

WINDOW = 0.2 #sec
FRAMES_PER_SECOND = 20
NUMBER_OF_PERMUTATIONS = 50
ALPHA = 0.0001

def plot_all_SCE_segments(segments, SCE_masks, p_r_s):
    number_of_segments = len(segments)
    for i in range(number_of_segments):
        plot_SCE_covarage(segments[i][0], SCE_masks[i], p_r_s)
        print i

    return

def plot_SCE_covarage(segment, SCE_mask, p_r_s):
    frame_rate = 1 / float(FRAMES_PER_SECOND)
    frames_per_window = WINDOW / frame_rate
    number_of_possible_SCE = len(SCE_mask)

    for frame in range(number_of_possible_SCE):
        if SCE_mask[frame]:
            SCE_activity = segment[:, frame:frame+frames_per_window]
            active_neurons = np.sum(SCE_activity, axis=1) > 0
            number_of_active_neurons = sum(active_neurons)
            # f, axx = subplots(number_of_active_neurons+1, 2, sharex=True)
            gs = gridspec.GridSpec(number_of_active_neurons+1, 3)
            ax0 = subplot(gs[:,0])
            ax0.matshow(SCE_activity, interpolation='none', aspect='auto')
            relevant_p_r_s = {}
            relevant_p_r_s['forward'] = p_r_s['forward'][active_neurons, :]
            relevant_p_r_s['backward'] = p_r_s['backward'][active_neurons, :]
            for nrn in range(number_of_active_neurons):
                if nrn==0:
                    ax = subplot(gs[nrn, 1])
                else:
                    ax = subplot(gs[nrn, 1], sharex=ax)
                ax.plot(relevant_p_r_s['forward'][nrn, :])
                ax.set_yticks([0, max(relevant_p_r_s['forward'][nrn, :])])
                ax = subplot(gs[nrn, 2], sharex=ax)
                ax.plot(relevant_p_r_s['backward'][nrn, :])
                ax.set_yticks([0, max(relevant_p_r_s['backward'][nrn, :])])

            ax = subplot(gs[number_of_active_neurons, 1])
            ax.plot(np.sum(relevant_p_r_s['forward'], axis=0), 'r')
            ax.set_yticks([0, max(np.sum(relevant_p_r_s['forward'], axis=0))])
            ax = subplot(gs[number_of_active_neurons, 2])
            ax.plot(np.sum(relevant_p_r_s['backward'], axis=0), 'r')
            ax.set_yticks([0, max(np.sum(relevant_p_r_s['backward'], axis=0))])
            show()
            raw_input('press enter to continue')
            close()
    return

def find_SCE_in_segments(segments, chance_activation):
    SCE_masks = []
    for segment in segments:
        SCE_masks.append(find_SCE_in_full_epoch(segment[0], chance_activation))

    return SCE_masks

def find_SCE_in_full_epoch(events, chance_activation):
    # Find SCEs in events matrix that are above chance level in a sliding time
    # window across entire epoch
    number_of_events_per_window = count_events_in_sliding_window(events)
    SCE_mask = number_of_events_per_window > chance_activation

    return SCE_mask

def calculte_SCE_chance_level(events):
    # Shuffle the events time for each neuron separately, and calculate the
    # chance level for number of active cells within a time window
    # Recommendation: the input events for this function, should be from all
    # rest epochs

    events_count = []
    for i in range(NUMBER_OF_PERMUTATIONS):
        current_event_permutation = shuffle_events_times(events)
        events_count.extend(count_events_in_sliding_window(
                            current_event_permutation))

    hist, edges = np.histogram(events_count, normed=True)
    pdf = np.cumsum(hist)*(edges[1])
    chance_activation = edges[np.where(pdf >= 1-ALPHA)[0][0] + 1]
    return chance_activation

def count_events_in_sliding_window(events):
    frame_rate = 1 / float(FRAMES_PER_SECOND)
    frames_per_window = np.int(WINDOW / frame_rate)
    number_of_frames = events.shape[1]
    total_events_per_frame = np.sum(events>0, axis=0)
    try:
        total_events_per_window = np.zeros((number_of_frames - frames_per_window))
        for i in range(number_of_frames - frames_per_window):
            total_events_per_window[i] = np.sum(
                        total_events_per_frame[i:i+frames_per_window])

        return total_events_per_window
    except ValueError:
        return []

def shuffle_events_times(events):
    shuffled_events = np.zeros_like(events)
    for i, neuron_events in enumerate(events):
        shuffled_events[i, :] = np.random.permutation(neuron_events)

    return shuffled_events

def concatenate_segments(events_segments, segment_type):
    # Args:
        # events_segments as generated from
        # create_segments_for_run_epochs_and_edges_entire_session
        # segment type=0 is edge segment, segement_type=1 is run segment
    concatenated_segments = []
    for segment in events_segments:
        concatenated_segments.append(segment[segment_type])

    concatenated_segments = np.hstack(concatenated_segments)

    return concatenated_segments

def main():

    for i, mouse in enumerate(MOUSE):

        for day in DAYS:
            print CAGE[i], mouse, day
            print
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' %(CAGE[i], mouse, day, ENV)
            events, traces, movement_data, p_r_s = load_session_data(session_dir)
            print 'number of neurons:', p_r_s['forward'].shape[0]
            events_segments_before = \
                create_segments_for_run_epochs_and_edges_entire_session(events,
                                                            movement_data,
                                                            'before', [],
                                                            EDGE_BINS)

            concatenated_edge_segments = \
                concatenate_segments(events_segments_before, 0)

            chance_activation = \
                calculte_SCE_chance_level(concatenated_edge_segments)

            print 'Chance activation:', chance_activation
            SCE_masks = find_SCE_in_segments(events_segments_before, chance_activation)

            plot_all_SCE_segments(events_segments_before, SCE_masks, p_r_s)

main()
