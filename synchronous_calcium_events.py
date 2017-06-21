# this module finds and make statistical calculations on synchronous calcium
# events (SCEs)

import numpy as np

from edges_events_probability_analysis import *

WINDOW = 200 #ms
FRAMES_PER_SECOND = 20
NUMBER_OF_PERMUTATIONS = 1
ALPHA = 0.05

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

    hist, edges = np.histogram(events_count, bins=20)
    pdf = np.cumsum(hist)
    chance_activation = edges(np.where(pdf >= 1-ALPHA)[0][0])

    return chance_activation

def count_events_in_sliding_window(events):
    frame_rate = 1 / FRAMES_PER_SECOND
    frames_per_window = WINDOW / frame_rate
    number_of_frames = events.shape[1]
    total_events_per_frame = np.sum(events, axis=0)
    total_events_per_window = np.zeros((number_of_frames - frames_per_window))
    for i in range(number_of_frames - frames_per_window):
        total_events_per_window[i] = np.sum(
                    total_events_per_frame[i:i+frames_per_window])

    return total_events_per_window

def shuffle_events_times(events):
    shuffled_events = np.zeros_like(events)
    for i, neuron_events in enumerate(events):
        shuffled_events[i, :] = np.random.shuffle(neuron_events)

    return shuffled_events

def concatenate_segments(events_segments, segment_type):
    # Args:
        # events_segments as generated from
        # create_segments_for_run_epochs_and_edges_entire_session
        # segment type=0 is edge segment, segement_type=1 is run segment
    concatenated_segments = []
    for segment in events_segments:
        concatenated_segments.extend(segment[segment_type])

    concatenated_segments = np.vstack(concatenated_segments)

    return concatenated_segments

def main():
    summary_figure, sum_ax = subplots(2, 3, sharex=True, sharey=True)

    for i, mouse in enumerate(MOUSE):
        mouse_name = 'c%dm%d' %(CAGE[i], mouse)

        for day in DAYS:
            print CAGE[i], mouse, day
            print
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' %(CAGE[i], mouse, day, ENV)
            events, traces, movement_data, _ = load_session_data(session_dir)
            events_segments_before = \
                create_segments_for_run_epochs_and_edges_entire_session(events,
                                                            movement_data,
                                                            'before', [],
                                                            EDGE_BINS)

            concatenated_edge_segments = \
                concatenate_segments(events_segments_before, 0)

            chance_activation = \
                calculte_SCE_chance_level(concatenated_edge_segments)

            find_SCE_in_segments(events_segments_before, chance_activation)



