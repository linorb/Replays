# this module finds and make statistical calculations on synchronous calcium
# events (SCEs)

import numpy as np
from matplotlib.pyplot import *
import matplotlib.gridspec as gridspec

from edges_events_probability_analysis import \
    create_segments_for_run_epochs_and_edges_entire_session,\
    load_session_data, calculate_conditional_activity_probability,\
    EDGE_BINS, MOUSE, CAGE, ENV, DAYS, WORK_DIR

WINDOW = 0.2 #sec
FRAMES_PER_SECOND = 20
NUMBER_OF_PERMUTATIONS = 50
ALPHA = 0.05
HIGH_PRECENTAGE= 0.1 # for conditional probability in edges - to find those cells that
            # have the high %HIGH_PRECENTAGE percent activation in edge given activation in run



def plot_all_SCE_segments(segments, SCE_masks, p_r_s):
    number_of_segments = len(segments)
    for i in range(number_of_segments):
        plot_SCE_covarage(segments[i][0], SCE_masks[i], p_r_s)
        print i

    return

def count_SCE_participation_per_neuron(segment, SCE_mask):
    # Count the number of SCE participation in segment
    frame_rate = 1 / float(FRAMES_PER_SECOND)
    frames_per_window = WINDOW / frame_rate
    number_of_frames = len(SCE_mask)
    SCE_counter = np.zeros(segment[0].shape[0])
    for frame in range(number_of_frames):
        if SCE_mask[frame]:
            SCE_activity = segment[0][:, frame:frame+frames_per_window]
            SCE_counter += np.sum(SCE_activity>0, axis=1)

    return SCE_counter

def count_SCE_participation_in_all_segments(segments, SCE_masks):
    SCE_counter =[]
    number_of_segments = len(segments)
    for i in range(number_of_segments):
        SCE_counter.append(count_SCE_participation_per_neuron(segments[i],
                                                              SCE_masks[i]))
    SCE_counter = np.vstack(SCE_counter)
    number_of_SCE_activations = np.sum(SCE_counter, axis=0)

    return number_of_SCE_activations

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
    for i, segment in enumerate(segments):
        SCE_masks.append(find_SCE_in_full_epoch(segment[0], chance_activation))

    return SCE_masks

def find_SCE_in_full_epoch(events, chance_activation):
    # Find SCEs in events matrix that are above chance level in a sliding time
    # window across entire epoch
    number_of_events_per_window = count_events_in_sliding_window(events)

    # In cases where the events have lower number of frames then the window size,
    # number_of_events_per_window returns [], and so this is what should be
    # returned from this function as well
    try:
        SCE_mask = number_of_events_per_window > chance_activation

        # Find only the unique SCEs (those which contain the maximum cells)
        # This is meant to deal with the running window summation of events count

        # Mask the SCE frames
        SCE_events_masked = np.ma.array(number_of_events_per_window, mask=SCE_mask)
        # Find the contiguous regions of SCEs
        SCE_regions = np.ma.clump_masked(SCE_events_masked)

        SCE_unique_mask = np.zeros_like(SCE_mask)
        for region in SCE_regions:
            # Inside a contiguous region, find the maximum number of events, and
            # mark him as True
            max_events_index = np.argmax(number_of_events_per_window[region])
            SCE_unique_mask[region][max_events_index] = True

        return SCE_unique_mask
    except IndexError:
        return []


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
    SCE_counts_all_mice = []
    p_edge_run_all_mice = []
    for i, mouse in enumerate(MOUSE):

        for day in DAYS:
            print CAGE[i], mouse, day
            print
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' %(CAGE[i], mouse, day, ENV)
            events, traces, movement_data, _, p_r_s = load_session_data(session_dir)
            print 'number of neurons:', events[0].shape[0]
            events_segments_before = \
                create_segments_for_run_epochs_and_edges_entire_session(events,
                                                            movement_data,
                                                            'before', [],
                                                            EDGE_BINS)
            # Calculate the conditional probability to be active in edge given
            # activity in run, and the highest activity level
            p_edge_run_before, _ = \
                calculate_conditional_activity_probability(
                    events_segments_before)
            p_edge_run_all_mice.append(p_edge_run_before)
            # Calculate the distribution of p(active in edge|active in run)
            hist_probability, edges_probability = np.histogram(p_edge_run_before[
                                           ~np.isnan(p_edge_run_before)],
                                       normed=True)
            pdf = np.cumsum(hist_probability) * (edges_probability[1])
            high_conditional_probability = edges_probability[
                np.where(pdf >= 1 - HIGH_PRECENTAGE)[0][0] + 1]

            high_probability_edge_neurons = p_edge_run_before > \
                                                  high_conditional_probability

            # Find SCE and the neurons that participate in it
            concatenated_edge_segments = \
                concatenate_segments(events_segments_before, 0)

            chance_SCE_activation = \
                calculte_SCE_chance_level(concatenated_edge_segments)

            print 'Chance SCE activation:', chance_SCE_activation
            SCE_masks = find_SCE_in_segments(events_segments_before,
                                             chance_SCE_activation)

            SCE_counts = \
                count_SCE_participation_in_all_segments(events_segments_before,
                                                        SCE_masks)
            SCE_counts_all_mice.append(SCE_counts)
            hist_SCE, edges_SCE = np.histogram(SCE_counts, normed=True)
            pdf = np.cumsum(hist_SCE) * (edges_SCE[1])
            high_SCE_participation = edges_SCE[
                np.where(pdf >= 1 - HIGH_PRECENTAGE)[0][0] + 1]

            high_SCE_participation_neurons = SCE_counts > high_SCE_participation

            # figure for histograms and SCE Vs. conditional probability
            # f, axx = subplots(3, 2)
            # axx[0, 0].bar(edges_probability[1:], hist_probability, width=0.07)
            # axx[0, 0].set_title('Histogram of p(edge|run)')
            # axx[0, 1].plot(high_probability_edge_neurons)
            # axx[0, 1].set_title('High conditional probability activation in edge')
            # axx[1, 0].bar(edges_SCE[1:], hist_SCE)
            # axx[1, 0].set_title('Histogram of SCE participant')
            # axx[1, 1].plot(high_SCE_participation_neurons)
            # axx[1, 1].set_title('high SCE activation')
            # combined = np.logical_and(high_probability_edge_neurons,
            #                               high_SCE_participation_neurons)
            # axx[2, 1].plot(combined)
            # axx[2, 1].set_title('combined')
            # print 'number of combined:', np.sum(combined)
            # axx[2, 0].plot(SCE_counts, p_edge_run_before, '*')
            # axx[2, 0].set_title('SCE counts Vs. p(edge|run)')
            # f.show()

            # plot_all_SCE_segments(events_segments_before, SCE_masks, p_r_s)

    SCE_counts_all_mice = np.concatenate(SCE_counts_all_mice)
    p_edge_run_all_mice =np.concatenate(p_edge_run_all_mice)
    relevent_indices = ~np.isnan(p_edge_run_all_mice)

    f, axx = subplots(3, 1)
    axx[0].hist(p_edge_run_all_mice[relevent_indices])
    axx[0].set_title('p(edge|run) histogram')
    axx[1].hist(SCE_counts_all_mice)
    axx[1].set_title('SCE counts histogram')
    axx[2].plot(SCE_counts_all_mice[relevent_indices],
                p_edge_run_all_mice[relevent_indices], '*')
    axx[2].set_title('SCE counts Vs p(edge|run)')
    f.suptitle('All mice neurons')
    f.show()

    raw_input('press enter to quit')

if __name__ == '__main__':
    main()
