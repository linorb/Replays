import numpy as np
import os
from matplotlib.pyplot import *
import scipy.stats

from bambi.tools import matlab
from bambi.tools.activity_loading import *
from bambi.analysis import maximum_likelihood
from zivlab.analysis.place_cells import find_place_cells

# Linear track parameters
FRAME_RATE = [20]*4
FRAME_RATE.extend([10]*6) #Hz
MOUSE = [4, 4, 1, 1, 6, 3, 6, 3, 0]
CAGE = [7, 6, 11, 13, 40, 40, 38, 38, 38]
ENV = [r'\envA']*4
ENV.extend([r'\linear']*6)
WORK_DIR = [r'D:\dev\replays\work_data\two_environments']*4
WORK_DIR.extend([r'D:\dev\replays\work_data\recall']*6)


# L-shape track parameters
# FRAME_RATE = [20]*4 #Hz
# MOUSE = [4, 4, 1, 1]
# CAGE = [6, 7, 11, 13]
# ENV= [r'\envB']*4
# WORK_DIR = [r'D:\dev\replays\work_data\two_environments']*4

EDGE_BINS = [0, 1, 10, 11]
VELOCITY_THRESHOLD = 5
NUMBER_OF_BINS = 24
SPACE_BINNING = 2
NUMBER_OF_PERMUTATIONS = 1
FRAMES_TO_LOOK_BACK = 15
MIN_NUMBER_OF_EVENTS = 15

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
    statistics['envA']['overall_decoding_fraction'] = sum(
        decoded_env == 0)/np.float32(len(decoded_env))
    statistics['envB']['overall_decoding_fraction'] = sum(
        decoded_env == 1) / np.float32(len(decoded_env))
    statistics['bucketA']['overall_decoding_fraction'] = sum(
        decoded_env == 2) / np.float32(len(decoded_env))
    statistics['bucketB']['overall_decoding_fraction'] = sum(
        decoded_env == 3) / np.float32(len(decoded_env))

    number_of_edge_binsA, _ = count_edge_bins(decoded_bins[decoded_env == 0],
                                              edge_bins)
    statistics['envA']['edge_decoding_fraction'] = \
        number_of_edge_binsA/np.float32(len(decoded_bins[decoded_env == 0]))

    number_of_edge_binsB, _ = count_edge_bins(decoded_bins[decoded_env == 1],
                                              edge_bins)
    statistics['envB']['edge_decoding_fraction'] = \
        number_of_edge_binsB/np.float32(len(decoded_bins[decoded_env == 1]))

    return statistics, decoded_bins, decoded_env

def calculate_p_val_for_correct_decoding_trial(events, p_neuron_bin, edge_bins,
                                               correct_decoding_percentage,
                                         number_of_permutations, environment):
    decoding_fraction = np.zeros((number_of_permutations))
    edge_fraction = np.zeros((number_of_permutations))
    for i in range(number_of_permutations):
        # shuffling neurons identities
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
                                               events, axx):
    number_of_events_per_frame = np.sum(events > 0, axis=0)
    number_of_events_per_frame = \
        number_of_events_per_frame[number_of_events_per_frame > 0]

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

def create_p_neuron_bin(movement_data, events_traces,
                        trials_indices):
    [bins, events] = create_training_data \
        (movement_data, events_traces, trials_indices)
    bins = wide_binning(bins, NUMBER_OF_BINS, SPACE_BINNING)

    p_neuron_bin = maximum_likelihood.create_delayed_p_r_s \
        (bins, events, FRAMES_TO_LOOK_BACK)

    return p_neuron_bin

def plot_median_error(median_error, mouse_name):

    mean_of_median_error = [np.mean(np.array(x)) for x in median_error]
    std_of_median_error = [np.std(np.array(x)) for x in median_error]
    number_of_sessions = len(median_error)

    f = figure()
    errorbar(range(number_of_sessions), mean_of_median_error, yerr=std_of_median_error)
    ylabel('Median error')
    xlabel('Session interval')
    title('Median error of maximum likelihood decoder\n' + mouse_name)
    f.show()


def main():

    for i, mouse in enumerate(MOUSE):
        # Creating a list of all mouse events and behavioral data
        mouse_events = []
        mouse_movement = []
        mouse_place_cells = []

        cell_reg_filename = WORK_DIR[i]+ '\c%dm%d\cellRegistered_%s.mat' %(CAGE[i], mouse, ENV[i][1:])
        cell_registration = matlab.load_cell_registration(cell_reg_filename)
        mouse_dir = WORK_DIR[i] + '\c%dm%d' % (CAGE[i], mouse)
        days_list = [x[1] for x in os.walk(mouse_dir)][0]

        for day in days_list:
            print 'loading data set for',CAGE[i], mouse, day

            # load the session data
            session_dir = mouse_dir + '\%s\%s' %(day, ENV[i])
            session_ind = int(day[-1])
            events_traces, movement_data = \
                load_session_data(session_dir, cell_registration, session_ind)

            mouse_events.append(events_traces)
            mouse_movement.append(movement_data)

            # find the place cells of current session (differencing directions):
            linear_trials_indices = range(len(events_traces))[1:-1]
            [bins, events] = create_training_data\
                (movement_data, events_traces, linear_trials_indices)

            bins = wide_binning(bins, NUMBER_OF_BINS, SPACE_BINNING)
            velocity = concatenate_movment_data\
                (movement_data, 'velocity', linear_trials_indices)

            velocity_positive = velocity > VELOCITY_THRESHOLD
            velocity_negative = velocity < -VELOCITY_THRESHOLD
            place_cells_positive, _, _ = find_place_cells\
                (bins[velocity_positive], events[:, velocity_positive],
                 min_number_of_events=MIN_NUMBER_OF_EVENTS)

            place_cells_negative, _, _ = find_place_cells\
                (bins[velocity_negative], events[:, velocity_negative],
                 min_number_of_events=MIN_NUMBER_OF_EVENTS)

            place_cells = np.concatenate\
                ([place_cells_positive, place_cells_negative])

            place_cells = np.unique(place_cells)

            mouse_place_cells.append(place_cells)

        # Loop on all sessions, for train and test inside session and between
        # sessions

        median_error_all_sessions = [[] for j,_ in enumerate(days_list)]
        for train_session_ind, day in enumerate(days_list):
            print 'training on data set for', CAGE[i], mouse, day
            # Create p_neuron_bin with all session trials for testing with other
            # sessions - Notice! the probability is for all neurons! there is a
            # need to separate for the common place cells in different sessions
            train_movement_data = mouse_movement[train_session_ind]
            train_events_traces = mouse_events[train_session_ind]
            train_place_cells = mouse_place_cells[train_session_ind]
            linear_trials_indices = range(len(train_events_traces))[1:-1]

            p_neuron_bin = create_p_neuron_bin(train_movement_data,
                                               train_events_traces,
                                               linear_trials_indices)

            number_of_days = len(days_list)
            for test_session_ind in np.arange(train_session_ind, number_of_days):
                print 'testing on data set for', CAGE[i], mouse, days_list[test_session_ind]
                # The case below is a special case where the session that is
                # tested is the session that the decoder trained on. and so,
                # for each trial there is a need to leave the tested trial out
                # of the training trials
                if test_session_ind == train_session_ind:
                    test_trials_indices = linear_trials_indices
                    for test_trial in test_trials_indices:
                        train_trials_indices = range(len(train_events_traces))[1:-1]
                        train_trials_indices.remove(test_trial)
                        current_p_neuron_bin = create_p_neuron_bin(train_movement_data,
                                                       train_events_traces,
                                                       train_trials_indices)


                        current_p_neuron_bin = [x[train_place_cells, :] for x in current_p_neuron_bin]
                        [test_bins, test_events] = create_training_data(
                            train_movement_data, train_events_traces,
                            [test_trial])
                        test_bins = wide_binning(test_bins, NUMBER_OF_BINS,
                                                 SPACE_BINNING)
                        estimated_bins, _ = decode_entire_trial \
                            (test_events[train_place_cells, :],
                             current_p_neuron_bin, FRAMES_TO_LOOK_BACK)

                        active_frames = np.sum(test_events, axis=0) > 0
                        # calculating the median_error for the frames that had activity in,
                        # since the frames that didn't have activity, get the
                        # estimation of the previous frame
                        median_error_bins = np.median(test_bins[active_frames] -
                                                    estimated_bins[
                                                        active_frames])
                        median_error_all_sessions \
                            [np.abs(train_session_ind - test_session_ind)]. \
                            append(median_error_bins)
                # The case below is for different sessions for training and testing
                else:
                    test_movement_data = mouse_movement[test_session_ind]
                    test_events_traces = mouse_events[test_session_ind]
                    test_place_cells = mouse_place_cells[test_session_ind]
                    shared_place_cells = np.intersect1d(train_place_cells,
                                                   test_place_cells)
                    test_trials_indices = range(len(test_events_traces))[1:-1]
                    current_p_neuron_bin = [x[shared_place_cells, :] for x in p_neuron_bin]
                    for test_trial in test_trials_indices:
                        [test_bins, test_events] = create_training_data(
                            test_movement_data, test_events_traces, [test_trial])
                        test_bins = wide_binning(test_bins, NUMBER_OF_BINS,
                                                 SPACE_BINNING)
                        estimated_bins, _ = decode_entire_trial \
                            (test_events[shared_place_cells, :],
                             current_p_neuron_bin, FRAMES_TO_LOOK_BACK)

                        active_frames = np.sum(test_events, axis=0) > 0
                        median_error_bins = np.median(test_bins[active_frames] -
                                           estimated_bins[active_frames])
                        median_error_all_sessions\
                        [np.abs(train_session_ind - test_session_ind)].\
                            append(median_error_bins)

        plot_median_error(median_error_all_sessions, 'C%sM%s' % (CAGE[i], mouse))


if __name__ == '__main__':
    main()