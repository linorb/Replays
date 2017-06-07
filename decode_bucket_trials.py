import numpy as np
import os
from matplotlib.pyplot import *
import scipy.stats

from bambi.tools import matlab
from bambi.tools.activity_loading import *
from bambi.analysis import maximum_likelihood
from zivlab.analysis.place_cells import find_place_cells

EDGE_BINS = [0, 1, 10, 11]
FRAME_RATE = 20 #Hz
MOUSE = [4, 4, 1, 1]
CAGE = [6, 7, 11, 13]
ENV = ['envA', 'envB']
DAYS = '1234567'
WORK_DIR = r'D:\dev\replays\work_data\two_environments'
VELOCITY_THRESHOLD = 1
NUMBER_OF_BINS = 24
SPACE_BINNING = 2
NUMBER_OF_PERMUTATIONS = 500

def load_session_data(session_dir, cell_registration, session_index):
    # Load events, traces, and behavioral data (my_mvmt) for entire session
    events_filename = 'finalEventsMat.mat'
    log_filename = 'frameLog.csv'
    behavior_filename = 'my_mvmt.mat'

    all_events = matlab.load_events_file(os.path.join(session_dir, events_filename))
    frame_log = matlab.load_frame_log_file(os.path.join(session_dir,log_filename))
    movement_data = matlab.load_mvmt_file(os.path.join(session_dir,behavior_filename))

    #Convert neuron numbering to global nubering
    all_events = unite_sessions([all_events], [session_index], cell_registration)

    events = order_events_into_trials(all_events, frame_log)

    return events, movement_data

def count_edge_bins(bins, edge_bins):
    edge_bins_mask = np.zeros_like(bins, dtype=bool)
    for b in edge_bins:
        edge_bins_mask[bins == b] = True

    number_of_edge_bins = sum(edge_bins_mask)

    return number_of_edge_bins, edge_bins_mask

def test_bucket_trial(events, p_neuron_bin, edge_bins):
    # Decode by using maximum-likelihood decoder for two environments a bucket trial
    # and return the decoding results, parentage of decoding from each environment,
    # and the division to edge bins and rest of track
    number_of_frames = events.shape[1]

    decoded_bins = np.zeros((number_of_frames))
    decoded_env = np.zeros((number_of_frames))

    # Decode each frame in events:
    for frame in range(number_of_frames):
        if np.sum(events[:, frame]) > 0:
            decoded_bins[frame], environment_name = decode_most_likely_bin_and_environment(
                                                    np.expand_dims(events[:, frame], axis=1), p_neuron_bin)
            if 'envA' in environment_name:
                decoded_env[frame] = 0
            else:
                decoded_env[frame] = 1
        else:
            decoded_bins[frame] = np.nan
            decoded_env[frame] = np.nan

    #Calculate environment statistics:
    statistics = {'envA': {'overall_decoding_fraction': [],
                           'edge_decoding_fraction': []},
                  'envB': {'overall_decoding_fraction': [],
                           'edge_decoding_fraction': []}}
    decoded_env = decoded_env[~np.isnan(decoded_env)]
    decoded_bins = decoded_bins[~np.isnan(decoded_bins)]
    statistics['envA']['overall_decoding_fraction'] = sum(decoded_env == 0)/np.float32(len(decoded_env))
    statistics['envB']['overall_decoding_fraction'] = sum(decoded_env == 1) / np.float32(len(decoded_env))

    number_of_edge_binsA, _ = count_edge_bins(decoded_bins[decoded_env == 0], edge_bins)
    statistics['envA']['edge_decoding_fraction'] = number_of_edge_binsA/np.float32(len(decoded_bins[decoded_env == 0]))

    number_of_edge_binsB, _ = count_edge_bins(decoded_bins[decoded_env == 1], edge_bins)
    statistics['envB']['edge_decoding_fraction'] = number_of_edge_binsB/np.float32(len(decoded_bins[decoded_env == 1]))

    return statistics, decoded_bins, decoded_env

def calculate_p_val_for_correct_decoding_trial(events, p_neuron_bin, edge_bins, correct_decoding_percentage,
                                         number_of_permutations, environment):
    decoding_fraction = np.zeros((number_of_permutations))
    edge_fraction = np.zeros((number_of_permutations))
    for i in range(number_of_permutations):
        events_permutation = np.random.permutation(events)
        statistics, decoded_bins, decoded_env =  test_bucket_trial(events_permutation, p_neuron_bin, edge_bins)
        # plot_two_env_histogram(decoded_bins, decoded_env, 'random')
        decoding_fraction[i] = (statistics[environment]['overall_decoding_fraction'])
        edge_fraction[i] = (statistics[environment]['edge_decoding_fraction'])

    p_val = {}
    p_val['overall_decoding_fraction'] = \
        sum(decoding_fraction > correct_decoding_percentage['overall_decoding_fraction'])/np.float32(number_of_permutations)
    p_val['edge_decoding_fraction'] = \
        sum(edge_fraction > correct_decoding_percentage['edge_decoding_fraction'])/np.float32(number_of_permutations)

    return p_val

def plot_two_env_histogram(decoded_bins, decoded_env, correct_env_name):
    env_A_mask = decoded_env == 0
    env_B_mask = decoded_env == 1

    f, axx = subplots(1, 2, sharex=True, sharey=True)
    axx[0].hist(decoded_bins[env_A_mask], bins=11)
    axx[0].set_title('Histogram of bins environment A')
    axx[0].set_xlabel('bins')
    axx[0].set_ylabel('number of frames')

    axx[1].hist(decoded_bins[env_B_mask], bins=11)
    axx[1].set_title('Histogram of bins environment B')
    axx[1].set_xlabel('bins')
    axx[1].set_ylabel('number of frames')

    f.suptitle('Correct environment:%s' %correct_env_name)

    f.show()

    return

def main():
    for i, mouse in enumerate(MOUSE):
        cell_reg_filename = WORK_DIR + '\c%dm%d\cellRegisteredFixed.mat' %(CAGE[i], mouse)
        cell_registration = matlab.load_cell_registration(cell_reg_filename)
        correct_decoding_percentage = []
        edge_decoding_percentage = []
        p_val_correct = []
        p_val_edge = []
        for session_ind, day in enumerate(DAYS):
            print CAGE[i], mouse, day
            place_cells = []
            p_neuron_bin = {}
            # Create training data for environment A
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' %(CAGE[i], mouse, day, ENV[0])
            events_tracesA, movement_dataA = load_session_data(session_dir, cell_registration, 2*session_ind)
            linear_trials_indicesA = range(len(events_tracesA))[1:-1]
            bucket_trials_indicesA = [0, len(events_tracesA) - 1]
            [binsA, eventsA] = create_training_data(movement_dataA, events_tracesA, linear_trials_indicesA)
            # use only events of place cells:
            binsA = wide_binning(binsA, NUMBER_OF_BINS, SPACE_BINNING)
            velocityA = concatenate_movment_data(movement_dataA, 'velocity', linear_trials_indicesA)
            velocity_positive = velocityA > VELOCITY_THRESHOLD
            velocity_negative = velocityA < -VELOCITY_THRESHOLD
            place_cells_positive, _, _ = find_place_cells(binsA[velocity_positive], eventsA[:, velocity_positive])
            place_cells_negative, _, _ = find_place_cells(binsA[velocity_negative], eventsA[:, velocity_negative])
            place_cellsA = np.concatenate([place_cells_positive, place_cells_negative])
            place_cells.append(place_cellsA)

            # Create training data for environment B
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' % (CAGE[i], mouse, day, ENV[1])
            events_tracesB, movement_dataB = load_session_data(session_dir, cell_registration, 2*session_ind+1)
            linear_trials_indicesB = range(len(events_tracesB))[1:-1]
            bucket_trials_indicesB = [0, len(events_tracesB)-1]
            [binsB, eventsB] = create_training_data(movement_dataB, events_tracesB, linear_trials_indicesB)
            binsB = wide_binning(binsB, NUMBER_OF_BINS, SPACE_BINNING)
            velocityB = concatenate_movment_data(movement_dataB, 'velocity', linear_trials_indicesB)
            velocity_positive = velocityB > VELOCITY_THRESHOLD
            velocity_negative = velocityB < -VELOCITY_THRESHOLD
            place_cells_positive, _, _ = find_place_cells(binsB[velocity_positive], eventsB[:, velocity_positive])
            place_cells_negative, _, _ = find_place_cells(binsB[velocity_negative], eventsB[:, velocity_negative])
            place_cellsB = np.concatenate([place_cells_positive, place_cells_negative])
            place_cells.append(place_cellsB)

            place_cells = np.concatenate(place_cells)
            place_cells = np.unique(place_cells)

            # dividing into two directions - positive, negative
            p_neuron_binA_positive = maximum_likelihood.calculate_p_r_s_matrix(binsA[velocityA > VELOCITY_THRESHOLD],
                                                                               eventsA[place_cells, :][:, velocityA >0])
            p_neuron_bin['envA_positive'] = p_neuron_binA_positive
            p_neuron_binA_negative = maximum_likelihood.calculate_p_r_s_matrix(binsA[velocityA < -VELOCITY_THRESHOLD],
                                                                               eventsA[place_cells, :][:, velocityA < 0])
            p_neuron_bin['envA_negative'] = p_neuron_binA_negative

            p_neuron_binB_positive = maximum_likelihood.calculate_p_r_s_matrix(binsB[velocityB > VELOCITY_THRESHOLD],
                                                                               eventsB[place_cells, :][:, velocityB > 0])
            p_neuron_bin['envB_positive'] = p_neuron_binB_positive
            p_neuron_binB_negative = maximum_likelihood.calculate_p_r_s_matrix(binsB[velocityB < -VELOCITY_THRESHOLD],
                                                                               eventsB[place_cells, :][:, velocityB < 0])
            p_neuron_bin['envB_negative'] = p_neuron_binB_negative

            # Testing bucket trials
            for trial in range(2):
                trial_events_A = events_tracesA[bucket_trials_indicesA[trial]][place_cells, :]
                statistics, decoded_bins, decoded_env = test_bucket_trial(trial_events_A, p_neuron_bin, EDGE_BINS)
                # plot_two_env_histogram(decoded_bins, decoded_env, 'A')
                correct_decoding_percentage.append(statistics['envA']['overall_decoding_fraction'])
                edge_decoding_percentage.append(statistics['envA']['edge_decoding_fraction'])
                p_val = calculate_p_val_for_correct_decoding_trial(trial_events_A, p_neuron_bin, EDGE_BINS,
                                                                   statistics['envA'],
                                                                    NUMBER_OF_PERMUTATIONS, 'envA')
                p_val_correct.append(p_val['overall_decoding_fraction'])
                p_val_edge.append(p_val['edge_decoding_fraction'])

                trial_events_B = events_tracesB[bucket_trials_indicesB[trial]][place_cells, :]
                statistics, decoded_bins, decoded_env = test_bucket_trial(trial_events_B, p_neuron_bin, EDGE_BINS)
                # plot_two_env_histogram(decoded_bins, decoded_env, 'B')
                correct_decoding_percentage.append(statistics['envB']['overall_decoding_fraction'])
                edge_decoding_percentage.append(statistics['envB']['edge_decoding_fraction'])
                p_val = calculate_p_val_for_correct_decoding_trial(trial_events_A, p_neuron_bin, EDGE_BINS,
                                                                   statistics['envB'],
                                                                   NUMBER_OF_PERMUTATIONS, 'envB')
                p_val_correct.append(p_val['overall_decoding_fraction'])
                p_val_edge.append(p_val['edge_decoding_fraction'])

        # plot all bucket trials env A and B
        f, axx = subplots(2, 2, sharex=True)
        # A bucket before
        axx[0, 0].bar(range(0, 28, 4), correct_decoding_percentage[0:28:4], color = 'blue')
        # A bucket after
        axx[0, 0].bar(range(1, 28, 4), correct_decoding_percentage[2:28:4], color = 'yellow')
        # B bucket before
        axx[0, 0].bar(range(2, 28, 4), correct_decoding_percentage[1:28:4], color = 'blue')
        # B bucket after
        axx[0, 0].bar(range(3, 28, 4), correct_decoding_percentage[3:28:4], color = 'yellow')
        axx[0, 0].plot(range(28), [0.5]*28, color = 'red')
        axx[0, 0].set_title('correct decoding fraction c%sm%s' %(CAGE[i], mouse))
        axx[0, 0].set_ylabel('fraction')
        axx[0, 0].set_ylim((0,1.1))
        axx[0, 0].set_xticks(np.arange(0.5, 27.5, 1))
        axx[0, 0].set_xticklabels(['A', 'A', 'B', 'B'] * 7)

        axx[1, 0].bar(range(0, 28, 4), p_val_correct[0:28:4], color='blue')
        axx[1, 0].bar(range(1, 28, 4), p_val_correct[2:28:4], color='yellow')
        axx[1, 0].bar(range(2, 28, 4), p_val_correct[1:28:4], color='blue')
        axx[1, 0].bar(range(3, 28, 4), p_val_correct[3:28:4], color='yellow')
        axx[1, 0].plot(range(28), [0.05] * 28, color='red')
        axx[1, 0].set_title('P value decoding fraction c%sm%s' % (CAGE[i], mouse))
        axx[1, 0].set_ylabel('fraction')
        axx[1, 0].set_ylim((0, 1.1))
        axx[1, 0].set_xticks(np.arange(0.5, 27.5, 1))
        axx[1, 0].set_xticklabels(['A', 'A', 'B', 'B'] * 7)

        axx[0, 1].bar(range(0, 28, 4), edge_decoding_percentage[0:28:4], color='blue')
        axx[0, 1].bar(range(1, 28, 4), edge_decoding_percentage[2:28:4], color='yellow')
        axx[0, 1].bar(range(2, 28, 4), edge_decoding_percentage[1:28:4], color='blue')
        axx[0, 1].bar(range(3, 28, 4), edge_decoding_percentage[3:28:4], color='yellow')
        axx[0, 1].plot(range(28), [0.5]*28, color = 'red')
        axx[0, 1].set_title('edge decoding fraction c%sm%s' % (CAGE[i], mouse))
        axx[0, 1].set_ylabel('fraction')
        axx[0, 1].set_ylim((0, 1.1))
        axx[0, 1].set_xticks(np.arange(0.5, 27.5, 1))
        axx[0, 1].set_xticklabels(['A', 'A', 'B', 'B'] * 7)

        axx[1, 1].bar(range(0, 28, 4), p_val_edge[0:28:4], color='blue')
        axx[1, 1].bar(range(1, 28, 4), p_val_edge[2:28:4], color='yellow')
        axx[1, 1].bar(range(2, 28, 4), p_val_edge[1:28:4], color='blue')
        axx[1, 1].bar(range(3, 28, 4), p_val_edge[3:28:4], color='yellow')
        axx[1, 1].plot(range(28), [0.05] * 28, color='red')
        axx[1, 1].set_title('P value fraction c%sm%s' % (CAGE[i], mouse))
        axx[1, 1].set_ylabel('fraction')
        axx[1, 1].set_ylim((0, 1.1))
        axx[1, 1].set_xticks(np.arange(0.5, 27.5, 1))
        axx[1, 1].set_xticklabels(['A', 'A', 'B', 'B'] * 7)

        f.show()

        f1, axx1 = subplots(1, 2, sharey=True, sharex=True)
        axx1[0].plot(p_val_correct[0:28:4], correct_decoding_percentage[0:28:4], 'ro', label='A first')
        axx1[0].plot(p_val_correct[1:28:4], correct_decoding_percentage[1:28:4], 'bo', label='B first')
        axx1[0].plot(p_val_correct[2:28:4], correct_decoding_percentage[2:28:4], 'go', label='A last')
        axx1[0].plot(p_val_correct[3:28:4], correct_decoding_percentage[3:28:4], 'ko', label='B last')
        axx1[0].set_ylim((-0.1, 1.1))
        axx1[0].set_xlim((-0.1, 1.1))
        axx1[0].set_title('P value Vs. correct decoding fraction c%sm%s' % (CAGE[i], mouse))
        axx1[0].set_ylabel('correct decoding fraction')
        axx1[0].set_xlabel('P value')
        axx1[0].legend(loc="upper right")


        axx1[1].plot(p_val_edge[0:28:4], edge_decoding_percentage[0:28:4], 'ro', label='A first')
        axx1[1].plot(p_val_edge[1:28:4], edge_decoding_percentage[1:28:4], 'bo', label='B first')
        axx1[1].plot(p_val_edge[2:28:4], edge_decoding_percentage[2:28:4], 'go', label='A last')
        axx1[1].plot(p_val_edge[3:28:4], edge_decoding_percentage[3:28:4], 'ko', label='B last')
        axx1[1].set_ylim((-0.1, 1.1))
        axx1[1].set_xlim((-0.1, 1.1))
        axx1[1].set_title('P value Vs. edge decoding fraction c%sm%s' % (CAGE[i], mouse))
        axx1[1].set_ylabel('edge decoding fraction')
        axx1[1].set_xlabel('P value')
        axx1[1].legend(loc="upper right")

        f1.show()
        #
    raw_input('press enter')



main()