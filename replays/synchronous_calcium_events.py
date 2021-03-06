# this module finds and make statistical calculations on synchronous calcium
# events (SCEs)

import numpy as np
import os
import matplotlib.gridspec as gridspec

from matplotlib.pyplot import *
from edges_events_probability_analysis import \
    create_segments_for_run_epochs_and_edges_entire_session,\
    load_session_data,EDGE_BINS

from bambi.tools.activity_loading import decode_entire_trial

# Linear track parameters
# FRAME_RATE = [10]*5 #Hz
# FRAME_RATE.extend([20]*4)
# MOUSE = [3, 6, 6, 3, 0, 4, 4, 1, 1]
# CAGE = [40, 40, 38, 38, 38, 6, 7, 11, 13]
# ENV = [r'\linear']*5
# ENV.extend([r'\envA']*4)
# WORK_DIR = [r'D:\dev\replays\work_data\recall']*5
# WORK_DIR.extend([r'D:\dev\replays\work_data\two_environments']*4)

# L shape parameters
FRAME_RATE = [20]*4 #Hz
MOUSE = [4, 4, 1, 1]
CAGE = [6, 7, 11, 13]
ENV= [r'\envB']*4
WORK_DIR = [r'D:\dev\replays\work_data\two_environments']*4

WINDOW = 0.2 #sec
NUMBER_OF_PERMUTATIONS = 500
ALPHA = 0.05
HIGH_PRECENTAGE= 0.1 # for conditional probability in edges - to find those cells that
            # have the high %HIGH_PRECENTAGE percent activation in edge given activation in run

def plot_all_SCE_segments(segments, SCE_masks, frame_rate, mouse_name):
    number_of_segments = len(segments)
    for i in range(number_of_segments):
        plot_segment_activity(segments[i], SCE_masks[i], frame_rate, mouse_name)
    return

def count_neurons_in_all_SCEs(segments, SCE_masks, frame_rate):

    neurons_counter_all_SCE = []
    fraction_of_run_all_SCE = []
    for segment, SCE_mask in zip(segments, SCE_masks):
        neurons_counter, fraction_of_run = \
            count_neurons_in_SCE(segment, SCE_mask, frame_rate)
        neurons_counter_all_SCE.append(neurons_counter)
        fraction_of_run_all_SCE.append(fraction_of_run)

    neurons_counter_all_SCE = np.concatenate(neurons_counter_all_SCE)
    fraction_of_run_all_SCE = np.concatenate(fraction_of_run_all_SCE)

    return neurons_counter_all_SCE, fraction_of_run_all_SCE

def count_neurons_in_SCE(segment, SCE_mask, frame_rate):
    # Count the number of neurons participate in SCE. and the fraction among
    # them that is active also in run segment
    frames_per_window = WINDOW * frame_rate
    number_of_frames = len(SCE_mask)
    neurons_counter = []
    count_run = []
    active_neurons_in_run = np.argwhere(np.sum(segment[1], axis=1) > 0)
    for frame in range(number_of_frames):
        if SCE_mask[frame]:
            SCE_activity = segment[0][:, frame:frame + frames_per_window]
            number_of_active_neurons_in_SCE = np.sum(np.sum(SCE_activity>0, axis=1))
            number_of_active_neurons_in_SCE_and_run =\
                np.sum(np.sum(SCE_activity[active_neurons_in_run, :]>0, axis=1))

            neurons_counter.append(number_of_active_neurons_in_SCE)
            count_run.append(number_of_active_neurons_in_SCE_and_run)

    return neurons_counter, count_run

def count_SCE_participation_per_neuron(segment, SCE_mask, frame_rate):
    # Count the number of SCE participation in segment
    frames_per_window = WINDOW * frame_rate
    number_of_frames = len(SCE_mask)
    SCE_counter = np.zeros(segment[0].shape[0])
    for frame in range(number_of_frames):
        if SCE_mask[frame]:
            SCE_activity = segment[0][:, frame:frame+frames_per_window]
            SCE_counter += np.sum(SCE_activity>0, axis=1)

    return SCE_counter

def count_SCE_participation_in_all_segments(segments, SCE_masks, frame_rate):
    SCE_counter =[]
    number_of_segments = len(segments)
    for i in range(number_of_segments):
        SCE_counter.append(count_SCE_participation_per_neuron(
                            segments[i], SCE_masks[i], frame_rate))
    SCE_counter = np.vstack(SCE_counter)
    number_of_SCE_activations = np.sum(SCE_counter, axis=0)

    return number_of_SCE_activations

def plot_segment_activity(segment, SCE_mask, frame_rate, mouse_name):
    # segmet is a list size 2: segment[0] is edge epoch, segment[1] is run epoch

    frames_per_window = int(WINDOW * frame_rate)
    number_of_possible_SCE = len(SCE_mask)

    # f1, axx1 = subplots(2, 1, sharex=True)
    gs = gridspec.GridSpec(3, 1)
    axx1 = []
    axx1.append(subplot(gs[0:2, :]))
    axx1.append(subplot(gs[2, :]))

    active_cells = np.sum(segment[0], axis=1) > 0
    axx1[0].matshow(segment[0][active_cells, :] > 0,
                   interpolation='none', aspect='auto', cmap='gray')
    axx1[0].set_title('Edge activity', fontsize=25)
    axx1[0].set_ylabel('Neuron number', fontsize=25)
    axx1[1].plot(np.sum(segment[0][active_cells, :] > 0, axis=0))
    axx1[1].set_xlabel('Time', fontsize=22)
    axx1[1].set_ylabel('Summed activity', fontsize=25)


    for frame in range(number_of_possible_SCE):
        if SCE_mask[frame]:
            axx1[0].axvline(frame, color='red')
            axx1[0].axvline(frame + frames_per_window, color='red')

            SCE_activity = segment[0][:, frame:frame + frames_per_window]
            active_neurons = np.sum(SCE_activity, axis=1) > 0
            run_activity = np.sum(segment[1][active_neurons, :] > 0)
            if run_activity > 5 | True:
                ind_neuron_sort = np.argsort(
                    np.argmax(segment[1][active_neurons, :], axis=1))
                f, axx = subplots(1, 2, sharey=True)
                axx[0].matshow(
                    SCE_activity[active_neurons, :][ind_neuron_sort, :] > 0,
                    interpolation='none', aspect='auto', cmap='gray')
                axx[0].set_title('SCE activity')
                axx[1].matshow(
                    segment[1][active_neurons, :][ind_neuron_sort, :] > 0,
                    interpolation='none', aspect='auto', cmap='gray')
                axx[1].set_title('Run activity')
                f.suptitle(mouse_name)
                f.show()
                savefig(mouse_name + '.pdf')

    for i in range(2):
        for xtick in axx1[i].xaxis.get_major_ticks():
            xtick.label2.set_fontsize(22)
            xtick.label1.set_fontsize(22)
        for ytick in axx1[i].yaxis.get_major_ticks():
            ytick.label.set_fontsize(22)
    return

def plot_SCE_covarage(segment, SCE_mask, p_r_s, frame_rate):
    # segment here is only the edge segment
    frames_per_window = WINDOW * frame_rate
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
            close()
    return

def find_SCE_in_segments(segments, chance_activation, frame_rate):
    SCE_masks = []
    for i, segment in enumerate(segments):
        SCE_masks.append(find_SCE_in_full_epoch(segment[0], chance_activation, frame_rate))

    return SCE_masks

def find_SCE_in_full_epoch(events, chance_activation, frame_rate):
    # Find SCEs in events matrix that are above chance level in a sliding time
    # window across entire epoch
    number_of_events_per_window = count_events_in_sliding_window(events, frame_rate)

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


def calculte_SCE_chance_level(events, frame_rate):
    # Calculate the SCE chance level by the top 5% of the events count in a window

    events_count = count_events_in_sliding_window(
        events, frame_rate)
    max_events = int(np.max(events_count))
    [count, bins] = np.histogram(events_count, bins=max_events+1)
    chance_activation = np.flip(bins)[np.argwhere(np.cumsum(np.flip(count)/float(np.sum(count))) < 0.05)[-1]+1]
    
    return chance_activation

def count_events_in_sliding_window(events, frame_rate):
    frames_per_window = np.int(WINDOW * frame_rate)
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

def plot_decoded_SCE_activity(segment, SCE_mask, frame_rate, place_cells, p_r_s):
    frames_per_window = WINDOW * frame_rate
    number_of_possible_SCE = len(SCE_mask)

    for frame in range(number_of_possible_SCE):
        if SCE_mask[frame]:
            SCE_activity = segment[0][:, frame:frame + frames_per_window]
            SCE_activity = SCE_activity[place_cells, :]
            decoded_activity, _ = decode_entire_trial(SCE_activity, p_r_s)
            f, axx = subplots(1,2,sharex=True)
            axx[0].matshow(SCE_activity, interpolation='none', aspect='auto')
            axx[0].set_title('SCE activity')
            axx[1].plot(decoded_activity)
            axx[1].set_title('Decoded activity')
            f.show()
    return

def main():
    neurons_counter_all_mice = []
    count_run_all_mice = []
    for i, mouse in enumerate(MOUSE):
        mouse_dir = WORK_DIR[i] + '\c%dm%d' % (CAGE[i], mouse)
        days_list = [x[1] for x in os.walk(mouse_dir)][0]
        for day in days_list:
            print CAGE[i], mouse, day
            print
            session_dir = mouse_dir + '\%s\%s' % (day, ENV[i])
            events, traces, movement_data, _, _ = load_session_data(session_dir)
            events_segments_before = \
                create_segments_for_run_epochs_and_edges_entire_session(events,
                                                            movement_data,
                                                            'before', [],
                                                            EDGE_BINS,
                                                            FRAME_RATE[i])

            # Find SCE and the neurons that participate in it
            concatenated_edge_segments = \
                concatenate_segments(events_segments_before, 0)

            chance_SCE_activation = \
                calculte_SCE_chance_level(concatenated_edge_segments, FRAME_RATE[i])

            SCE_masks = find_SCE_in_segments(
                events_segments_before,chance_SCE_activation, FRAME_RATE[i])
            plot_all_SCE_segments(events_segments_before, SCE_masks,
                                  FRAME_RATE[i],
                                  'C%dM%d environment %s session %s'
                                  % (CAGE[i], mouse, ENV[i][-1], day))

            neurons_counter, count_run = \
                count_neurons_in_all_SCEs(events_segments_before, SCE_masks,
                                          FRAME_RATE[i])

            neurons_counter_all_mice.append(neurons_counter)
            count_run_all_mice.append(count_run)

    neurons_counter_all_mice = np.concatenate(neurons_counter_all_mice)
    count_run_all_mice = np.concatenate(count_run_all_mice)

    relevant_indices = ~np.isnan(count_run_all_mice)

    # np.savez('SCE_analysis_Lshape', neurons_counter_all_mice = neurons_counter_all_mice,
    #          count_run_all_mice = count_run_all_mice,
    #          relevant_indices =relevant_indices)

    np.savez('SCE_analysis_Lshape',
             neurons_counter_all_mice=neurons_counter_all_mice,
             count_run_all_mice=count_run_all_mice,
             relevant_indices=relevant_indices)

    raw_input('press enter')

if __name__ == '__main__':
    main()
