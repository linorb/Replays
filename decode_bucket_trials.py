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
VELOCITY_THRESHOLD = 3
NUMBER_OF_BINS = 24
SPACE_BINNING = 2
NUMBER_OF_PERMUTATIONS = 1000


def load_session_data(session_dir, cell_registration, session_index):
    # Load events, traces, and behavioral data (my_mvmt) for entire session
    events_filename = 'finalEventsMat.mat'
    log_filename = 'frameLog.csv'
    if 'envA' in session_dir:
        behavior_filename = 'my_mvmt_smooth.mat'
    else: # envB
        behavior_filename = 'my_mvmt_fixed.mat'
        print behavior_filename

    all_events = matlab.load_events_file(os.path.join(session_dir, events_filename))
    frame_log = matlab.load_frame_log_file(os.path.join(session_dir,log_filename))
    movement_data = matlab.load_mvmt_file(os.path.join(session_dir,behavior_filename))

    #Convert neuron numbering to global nubering
    all_events = unite_sessions([all_events], [session_index], cell_registration)

    events = order_events_into_trials(all_events, frame_log)

    return events, movement_data


def create_bucket_training_data(events_divided_to_trials,
                                bucket_trials_indices):
    # create a training data for bucket trials according to the format that
    # fits the create_p_r_s in maximum likelihood module

    concatenated_bins = []
    for trial_index in bucket_trials_indices:
        number_of_frames_in_trial = \
        events_divided_to_trials[trial_index].shape[1]
        temp_array = np.zeros(number_of_frames_in_trial, dtype=np.int)
        concatenated_bins.append(temp_array)

    concatenated_bins = np.concatenate(concatenated_bins)

    concatenated_events = []
    for trial_index in bucket_trials_indices:
        concatenated_events.append(events_divided_to_trials[trial_index])

    concatenated_events = np.hstack(concatenated_events)

    return concatenated_bins, concatenated_events


def count_edge_bins(bins, edge_bins):
    edge_bins_mask = np.zeros_like(bins, dtype=bool)
    for b in edge_bins:
        edge_bins_mask[bins == b] = True

    number_of_edge_bins = sum(edge_bins_mask)

    return number_of_edge_bins, edge_bins_mask


def test_bucket_trial(events, p_neuron_bin, edge_bins):
    # Decode by using maximum-likelihood decoder for two environments a bucket
    #  trial and return the decoding results, parentage of decoding from each
    # environment, and the division to edge bins and rest of track
    number_of_frames = events.shape[1]

    decoded_bins = np.zeros((number_of_frames))
    decoded_env = np.zeros((number_of_frames))

    # Decode each frame in events:
    for frame in range(number_of_frames):
        if np.sum(events[:, frame]) > 0:
            decoded_bins[frame], environment_name = \
                decode_most_likely_bin_and_environment(
                np.expand_dims(events[:, frame], axis=1), p_neuron_bin)
            if 'envA' in environment_name:
                decoded_env[frame] = 0
            elif 'envB' in environment_name:
                decoded_env[frame] = 1
            elif 'bucketA' in environment_name:
                decoded_env[frame] = 2
            elif 'bucketB' in environment_name:
                decoded_env[frame] = 3
        else:
            decoded_bins[frame] = np.nan
            decoded_env[frame] = np.nan

    #Calculate environment statistics:
    statistics = {'envA': {'overall_decoding_fraction': [],
                           'edge_decoding_fraction': []},
                  'envB': {'overall_decoding_fraction': [],
                           'edge_decoding_fraction': []},
                  'bucketA': {'overall_decoding_fraction': []},
                  'bucketB': {'overall_decoding_fraction': []}
                  }
    decoded_env = decoded_env[~np.isnan(decoded_env)]
    decoded_bins = decoded_bins[~np.isnan(decoded_bins)]

    statistics['envA']['overall_decoding_fraction'] = \
        sum(decoded_env == 0)/np.float32(len(decoded_env))
    statistics['envB']['overall_decoding_fraction'] = \
        sum(decoded_env == 1) / np.float32(len(decoded_env))

    number_of_edge_binsA, _ = count_edge_bins(decoded_bins[decoded_env == 0],
                                              edge_bins)
    statistics['envA']['edge_decoding_fraction'] = \
        number_of_edge_binsA/np.float32(len(decoded_bins[decoded_env == 0]))

    number_of_edge_binsB, _ = count_edge_bins(decoded_bins[decoded_env == 1],
                                              edge_bins)
    statistics['envB']['edge_decoding_fraction'] = \
        number_of_edge_binsB/np.float32(len(decoded_bins[decoded_env == 1]))

    return statistics, decoded_bins, decoded_env


def calculate_p_val_for_correct_decoding_trial(events, p_neuron_bin,
                                               edge_bins,
                                               correct_decoding_percentage,
                                               number_of_permutations,
                                               environment):

    decoding_fraction = np.zeros((number_of_permutations))
    edge_fraction = np.zeros((number_of_permutations))
    for i in range(number_of_permutations):
        # shuffling neurons identities
        events_permutation = np.random.permutation(events)

        statistics, _, _ =  test_bucket_trial(events_permutation,
                                              p_neuron_bin, edge_bins)
        decoding_fraction[i] = (statistics[environment]\
                                    ['overall_decoding_fraction'])
        edge_fraction[i] = (statistics[environment]['edge_decoding_fraction'])

    p_val = {}
    p_val['overall_decoding_fraction'] = \
        sum(decoding_fraction > correct_decoding_percentage\
            ['overall_decoding_fraction'])/np.float32(number_of_permutations)
    p_val['edge_decoding_fraction'] = \
        sum(edge_fraction > correct_decoding_percentage\
            ['edge_decoding_fraction'])/np.float32(number_of_permutations)

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


def plot_decoded_bins(decoded_bins, decoded_env, correct_env_name):
    envA_bins = np.zeros_like(decoded_bins)
    envA_bins[:] = np.nan
    envB_bins = np.zeros_like(decoded_bins)
    envB_bins[:] = np.nan
    envA_bins[decoded_env == 0] = decoded_bins[decoded_env == 0]
    envB_bins[decoded_env == 1] = decoded_bins[decoded_env == 1]

    bucketA_bins = np.zeros_like(decoded_bins)
    bucketA_bins[:] = np.nan
    bucketB_bins = np.zeros_like(decoded_bins)
    bucketB_bins[:] = np.nan
    bucketA_bins[decoded_env == 2] = decoded_bins[decoded_env == 2]
    bucketB_bins[decoded_env == 3] = decoded_bins[decoded_env == 3]

    f, axx = subplots(4, 2, sharex='col', sharey='col')
    axx[0, 0].plot(envA_bins, '*')
    axx[0, 0].set_title('Decoded bins in environment A')
    axx[1, 0].plot(envB_bins, '*')
    axx[1, 0].set_title('Decoded bins in environment B')
    axx[2, 0].plot(bucketA_bins, '*')
    axx[2, 0].set_title('Decoded bins in bucket A')
    axx[3, 0].plot(bucketB_bins, '*')
    axx[3, 0].set_title('Decoded bins in bucket B')

    axx[1, 0].set_ylim((-0.1, 12))
    setp(axx[:, 0], xlabel='#frame', ylabel='#bin')

    edges = [0, 1, 2, 3, 4]
    axx[2, 1].hist(decoded_env[~np.isnan(decoded_env)], bins=edges, normed=True)
    axx[2, 1].set_title('probability for environment decoding')
    f.suptitle('Correct environment:%s' %(correct_env_name))

    f.show()

    return


def plot_decoded_bin_vs_number_of_active_cells(decoded_bins, decoded_env,
                                               number_of_events_per_frame,
                                               events, axx):


    axx[0].plot(decoded_bins[decoded_env == 0],
                number_of_events_per_frame[decoded_env == 0], '*')
    axx[0].set_xlim((-0.1, np.max(decoded_bins[decoded_env == 0]) + 1))
    # axx[0].set_ylim((0, np.max(number_of_events_per_frame[decoded_env == 0])+1))
    axx[0].set_title('Environment A')
    axx[0].set_xlabel('Decoded bins')
    axx[0].set_ylabel('Number of events in frame')

    axx[1].plot(decoded_bins[decoded_env == 1],
                number_of_events_per_frame[decoded_env == 1], '*')
    axx[1].set_xlim((-0.1, np.max(decoded_bins[decoded_env == 1])+1))
    # axx[1].set_ylim((0, np.max(number_of_events_per_frame[decoded_env == 1])+1))
    axx[1].set_title('Environment B')
    axx[1].set_xlabel('Decoded bins')
    axx[1].set_ylabel('Number of events in frame')

    return axx

def main():
    for i, mouse in enumerate(MOUSE):
        correct_decoding_percentage = []
        edge_decoding_percentage = []
        p_val_correct = []
        p_val_edge = []
        decoded_bins_all_sessions = {'envA': [], 'envB': []}
        decoded_env_all_sessions = {'envA': [], 'envB': []}
        number_of_events_per_frame_all_sessions = {'envA': [], 'envB': []}
        all_bins = {'envA': [], 'envB': []}
        all_velocity = {'envA': [], 'envB': []}
        cell_reg_filename = WORK_DIR + '\c%dm%d\cellRegisteredFixed.mat' \
                                       %(CAGE[i], mouse)
        cell_registration = matlab.load_cell_registration(cell_reg_filename)

        for session_ind, day in enumerate(DAYS):
            print CAGE[i], mouse, day
            place_cells = []
            p_neuron_bin = {}
            # Create training data for environment A
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s'\
                                     %(CAGE[i], mouse, day, ENV[0])
            events_tracesA, movement_dataA = load_session_data\
                (session_dir, cell_registration, 2*session_ind)
            linear_trials_indicesA = range(len(events_tracesA))[1:-1]
            bucket_trials_indicesA = [0, len(events_tracesA) - 1]
            [binsA, eventsA] = create_training_data\
                (movement_dataA, events_tracesA, linear_trials_indicesA)

            # use only events of place cells:
            binsA = wide_binning(binsA, NUMBER_OF_BINS, SPACE_BINNING)
            velocityA = concatenate_movment_data\
                (movement_dataA, 'velocity', linear_trials_indicesA)
            all_bins['envA'].append(binsA)
            all_velocity['envA'].append(velocityA)
            velocity_positive = velocityA > VELOCITY_THRESHOLD
            velocity_negative = velocityA < -VELOCITY_THRESHOLD
            place_cells_positive, _, _ = find_place_cells\
                (binsA[velocity_positive], eventsA[:, velocity_positive])

            place_cells_negative, _, _ = find_place_cells\
                (binsA[velocity_negative], eventsA[:, velocity_negative])

            place_cellsA = np.concatenate\
                ([place_cells_positive, place_cells_negative])
            #
            place_cells.append(place_cellsA)

            # Create training data for environment B
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' % \
                                     (CAGE[i], mouse, day, ENV[1])

            events_tracesB, movement_dataB = load_session_data\
                (session_dir, cell_registration, 2*session_ind+1)

            linear_trials_indicesB = range(len(events_tracesB))[1:-1]
            bucket_trials_indicesB = [0, len(events_tracesB)-1]
            [binsB, eventsB] = create_training_data\
                (movement_dataB, events_tracesB, linear_trials_indicesB)

            binsB = wide_binning(binsB, NUMBER_OF_BINS, SPACE_BINNING)
            velocityB = concatenate_movment_data\
                (movement_dataB, 'velocity', linear_trials_indicesB)
            all_bins['envB'].append(binsB)
            all_velocity['envB'].append(velocityB)
            velocity_positive = velocityB > VELOCITY_THRESHOLD
            velocity_negative = velocityB < -VELOCITY_THRESHOLD
            place_cells_positive, _, _ = find_place_cells\
                (binsB[velocity_positive], eventsB[:, velocity_positive])

            place_cells_negative, _, _ = find_place_cells\
                (binsB[velocity_negative], eventsB[:, velocity_negative])

            place_cellsB = np.concatenate\
                ([place_cells_positive, place_cells_negative])


            place_cells.append(place_cellsB)

            place_cells = np.concatenate(place_cells)
            place_cells = np.unique(place_cells)

            # dividing into two directions - positive, negative
            p_neuron_binA_positive = maximum_likelihood.calculate_p_r_s_matrix\
                (binsA[velocityA > VELOCITY_THRESHOLD],
                 eventsA[place_cells, :][:, velocityA >VELOCITY_THRESHOLD])
            p_neuron_bin['envA_positive'] = [p_neuron_binA_positive]

            p_neuron_binA_negative = maximum_likelihood.calculate_p_r_s_matrix\
                (binsA[velocityA < -VELOCITY_THRESHOLD],
                 eventsA[place_cells, :][:, velocityA < -VELOCITY_THRESHOLD])
            p_neuron_bin['envA_negative'] = [p_neuron_binA_negative]

            p_neuron_binB_positive = maximum_likelihood.calculate_p_r_s_matrix\
                (binsB[velocityB > VELOCITY_THRESHOLD],
                 eventsB[place_cells, :][:, velocityB > VELOCITY_THRESHOLD])
            p_neuron_bin['envB_positive'] = [p_neuron_binB_positive]

            p_neuron_binB_negative = maximum_likelihood.calculate_p_r_s_matrix\
                (binsB[velocityB < -VELOCITY_THRESHOLD],
                 eventsB[place_cells, :][:, velocityB < -VELOCITY_THRESHOLD])
            p_neuron_bin['envB_negative'] = [p_neuron_binB_negative]

            for trial in range(2):
                trial_events_A = events_tracesA[bucket_trials_indicesA[trial]]\
                [place_cells, :]
                statistics , decoded_bins, decoded_env = \
                    test_bucket_trial(trial_events_A, p_neuron_bin, EDGE_BINS)

                number_of_events_per_frame = np.sum(trial_events_A > 0, axis=0)
                number_of_events_per_frame = \
                    number_of_events_per_frame[number_of_events_per_frame > 0]

                decoded_bins_all_sessions['envA'].append(decoded_bins)
                decoded_env_all_sessions['envA'].append(decoded_env)
                number_of_events_per_frame_all_sessions['envA'].append\
                    (number_of_events_per_frame)

                correct_decoding_percentage.append\
                    (statistics['envA']['overall_decoding_fraction'])
                #
                # edge_decoding_percentage.append\
                #     (statistics['envA']['edge_decoding_fraction'])
                #
                # p_val = calculate_p_val_for_correct_decoding_trial\
                #     (trial_events_A, p_neuron_bin, EDGE_BINS, statistics['envA'],
                #     NUMBER_OF_PERMUTATIONS, 'envA')
                #
                # p_val_correct.append(p_val['overall_decoding_fraction'])
                # p_val_edge.append(p_val['edge_decoding_fraction'])

                trial_events_B = events_tracesB[bucket_trials_indicesB[trial]]\
                    [place_cells, :]

                statistics, decoded_bins, decoded_env  = \
                    test_bucket_trial(trial_events_B, p_neuron_bin, EDGE_BINS)

                number_of_events_per_frame = np.sum(trial_events_B > 0, axis=0)
                number_of_events_per_frame = \
                    number_of_events_per_frame[number_of_events_per_frame > 0]

                decoded_bins_all_sessions['envB'].append(decoded_bins)
                decoded_env_all_sessions['envB'].append(decoded_env)
                number_of_events_per_frame_all_sessions['envB'].append \
                    (number_of_events_per_frame)

                correct_decoding_percentage.append\
                    (statistics['envB']['overall_decoding_fraction'])
                #
                # edge_decoding_percentage.append\
                #     (statistics['envB']['edge_decoding_fraction'])
                #
                # p_val = calculate_p_val_for_correct_decoding_trial\
                #     (trial_events_A, p_neuron_bin, EDGE_BINS, statistics['envB'],
                #      NUMBER_OF_PERMUTATIONS, 'envB')
                #
                # p_val_correct.append(p_val['overall_decoding_fraction'])
                # p_val_edge.append(p_val['edge_decoding_fraction'])

        # np.savez('bucket_decoding_statistics_c%sm%s' %(CAGE[i], mouse),
        #          correct_decoding_percentage = correct_decoding_percentage,
        #          edge_decoding_percentage = edge_decoding_percentage,
        #          p_val_correct = p_val_correct,
        #          p_val_edge = p_val_edge)

        np.savez('bucket_decoding_results_c%sm%s' % (CAGE[i], mouse),
                 correct_decoding_percentage = correct_decoding_percentage,
                 decoded_bins_all_sessions=decoded_bins_all_sessions,
                 decoded_env_all_sessions=decoded_env_all_sessions,
                 number_of_events_per_frame_all_sessions=
                 number_of_events_per_frame_all_sessions)

        np.savez('bins_velocity_c%sm%s' % (CAGE[i], mouse), all_bins=all_bins,
                 all_velocity=all_velocity)
        #
        np.savez('p_neuron_bin_c%sm%s' % (CAGE[i], mouse),
                 p_neuron_bin=p_neuron_bin)

    raw_input('press enter')

if __name__ == '__main__':
    main()