import numpy as np
import os
from matplotlib.pyplot import *
import scipy.stats

from bambi.tools import matlab
from bambi.tools.activity_loading import *
from bambi.analysis import maximum_likelihood
from zivlab.analysis.place_cells import find_place_cells
from decode_bucket_trials import test_bucket_trial

# Linear track parameters
FRAME_RATE = [20]*4
FRAME_RATE.extend([10]*5) #Hz
MOUSE = [4, 4, 1, 1]
CAGE = [6, 7, 11, 13]
ENV = [r'\envA']*4
WORK_DIR = [r'D:\dev\replays\work_data\two_environments']*4

# L-shape track parameters
# FRAME_RATE = [20]*4 #Hz
# MOUSE = [4, 4, 1, 1]
# CAGE = [6, 7, 11, 13]
# ENV= [r'\envB']*4
# WORK_DIR = [r'D:\dev\replays\work_data\two_environments']*4

EDGE_BINS = [0, 1, 10, 11]
VELOCITY_THRESHOLD = 3
NUMBER_OF_BINS = 24
SPACE_BINNING = 2
NUMBER_OF_PERMUTATIONS = 100
FRAMES_TO_LOOK_BACK = 0
MIN_NUMBER_OF_EVENTS = 15

def load_session_data(session_dir, cell_registration, session_index):
    # Load events, traces, and behavioral data (my_mvmt) for entire session
    events_filename = 'fixedEventsMat.mat'
    log_filename = 'frameLog.csv'
    if 'envA' in session_dir:
        behavior_filename = 'my_mvmt_smooth.mat'
    else: # envB
        behavior_filename = 'my_mvmt_fixed.mat'

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

def calculate_p_val_for_mae(events, bins, p_neuron_bin,
                            original_mae, number_of_permutations):
    mae_permutation = np.zeros(number_of_permutations)
    for i in range(number_of_permutations):
        # shuffling neurons identities
        events_permutation = np.random.permutation(events)
        decoded_bins, _ = decode_entire_trial \
                            (events_permutation, p_neuron_bin,
                             FRAMES_TO_LOOK_BACK)
        active_frames = np.sum(events, axis=0) > 0
        mae_permutation[i] = np.mean(np.abs(decoded_bins[active_frames]-
                                             bins[active_frames]))

    p_val =  \
        sum(mae_permutation < original_mae)/np.float32(number_of_permutations)

    return p_val, mae_permutation


def decode_trial(trial_events, p_neuron_bin,
                        number_of_frames_to_look_back=1):
    """Decode most likely bin for each frame in the trial

    Args:
        trial_events: A matrix of neuronal activity from entire trial
        p_neuron_bin: A list of matrices of the probability P(active neuruon| current bin)
        number_of_frames_to_look_back: Number of frames to look back for more
                                        accurate estimation (10 is acceptable
                                        number)

    Returns:
        estimated_bins: An array of most likely bins of each frame in the trial

    """
    number_of_frames = trial_events.shape[1]
    frame_indices = range(number_of_frames_to_look_back, number_of_frames)
    estimated_bins = np.zeros([number_of_frames])
    estimated_prob = np.zeros([number_of_frames])
    for frame in frame_indices:
        neuronal_activity = \
            trial_events[:, frame - number_of_frames_to_look_back:frame + 1]
        neuronal_activity = neuronal_activity > 0
        [estimated_bins[frame], estimated_prob[frame]] = \
            estimate_maximum_likelihood \
                (neuronal_activity, p_neuron_bin)

    return estimated_bins, estimated_prob

def plot_decoded_bins(decoded_bins, real_bins, session_details):
    f = figure()
    line1, = plot(real_bins, 'b', label='actual behavior')
    line2, = plot(decoded_bins, 'r*', label='decoded behavior')
    legend(bbox_to_anchor=[1.1, 1.1], handles=[line1, line2], fontsize=20)
    xlabel('Time [Sec]', fontsize=22)
    ylabel('Position [bin]', fontsize=22)
    xticks(np.arange(0, 3500, 500), np.arange(0, 350, 50), fontsize=22)
    yticks(fontsize=22)
    f.show()

    return

def create_p_neuron_bin(movement_data, events_traces,
                        trials_indices):
    [bins, events] = create_training_data \
        (movement_data, events_traces, trials_indices)
    bins = wide_binning(bins, NUMBER_OF_BINS, SPACE_BINNING)

    p_neuron_bin = maximum_likelihood.create_delayed_p_r_s \
        (bins, events, FRAMES_TO_LOOK_BACK)

    return p_neuron_bin

def plot_mean_error(mean_error, mouse_name, path):

    mean_of_mean_error = [np.mean(np.array(x)) for x in mean_error]
    std_of_mean_error = [np.std(np.array(x)) for x in mean_error]
    number_of_sessions = len(mean_error)

    f = figure()
    errorbar(range(number_of_sessions), mean_of_mean_error, yerr=std_of_mean_error)
    ylabel('Mean error')
    xlabel('Session interval')
    title('mean error of maximum likelihood decoder\n' + mouse_name)
    f.show()
    savefig(path + r'\mean_error_decoder_' + mouse_name + '.pdf')

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
            session_dir = mouse_dir + '\%s%s' %(day, ENV[i])
            print session_dir
            session_ind = int(day[-1])-1
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

        mean_error_all_sessions = [[] for j,_ in enumerate(days_list)]
        pval_for_mean_error = [[] for j,_ in enumerate(days_list)]
        mean_error_permutaion_all_sessions = [[] for j,_ in enumerate(days_list)]
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
                        print test_trial
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
                        estimated_bins, _ = decode_trial \
                            (test_events[train_place_cells, :],
                             current_p_neuron_bin, FRAMES_TO_LOOK_BACK)

                        active_frames = np.sum(test_events[train_place_cells, :]
                                               , axis=0) > 0
                        # calculating the mean_error for the frames that had activity in,
                        # since the frames that didn't have activity, get the
                        # estimation of the previous frame
                        mean_error_bins = np.mean(np.abs((test_bins[active_frames] -
                                                    estimated_bins[
                                                        active_frames])))
                        # pval, mae_permutation = calculate_p_val_for_mae(test_events[train_place_cells, :],
                        #                         test_bins,
                        #                         current_p_neuron_bin,
                        #                         mean_error_bins,
                        #                         NUMBER_OF_PERMUTATIONS)

                        session_details = 'C%sM%s - train session: %d, ' \
                                          'test session+trial: %d %d\n mean error: %d'\
                                          %(CAGE[i], mouse, train_session_ind,
                                            test_session_ind, test_trial, float(mean_error_bins))
                        plot_decoded_bins(estimated_bins[active_frames],
                                          test_bins[active_frames],
                                          session_details)

                        # plot_decoded_bins(estimated_bins, test_bins,
                        #                   session_details)

                        # mean_error_all_sessions \
                        #     [np.abs(train_session_ind - test_session_ind)]. \
                        #     append(mean_error_bins)
                        # pval_for_mean_error \
                        #     [np.abs(train_session_ind - test_session_ind)]. \
                        #     append(pval)
                        # mean_error_permutaion_all_sessions \
                        #     [np.abs(train_session_ind - test_session_ind)]. \
                        #     append(mae_permutation)

                # The case below is for different sessions for training and testing
                # else:
                #     test_movement_data = mouse_movement[test_session_ind]
                #     test_events_traces = mouse_events[test_session_ind]
                #     test_place_cells = mouse_place_cells[test_session_ind]
                #     shared_place_cells = np.intersect1d(train_place_cells,
                #                                    test_place_cells)
                #     test_trials_indices = range(len(test_events_traces))[1:-1]
                #     current_p_neuron_bin = [x[shared_place_cells, :] for x in p_neuron_bin]
                #
                #     for test_trial in test_trials_indices:
                #         [test_bins, test_events] = create_training_data(
                #             test_movement_data, test_events_traces, [test_trial])
                #         test_bins = wide_binning(test_bins, NUMBER_OF_BINS,
                #                                  SPACE_BINNING)
                #         estimated_bins, _ = decode_trial \
                #             (test_events[shared_place_cells, :],
                #              current_p_neuron_bin, FRAMES_TO_LOOK_BACK)
                #
                #         active_frames = np.sum(test_events, axis=0) > 0
                #         mean_error_bins = np.mean(np.abs((test_bins[active_frames] -
                #                            estimated_bins[active_frames])))
                #         pval, mae_permutation = calculate_p_val_for_mae(
                #             test_events[shared_place_cells, :],
                #             test_bins,
                #             current_p_neuron_bin,
                #             mean_error_bins,
                #             NUMBER_OF_PERMUTATIONS)
                #
                #         session_details = 'C%sM%s - train session: %d, ' \
                #                           'test session+trial: %d %d\n mean error: %d' \
                #                           % (CAGE[i], mouse, train_session_ind,
                #                              test_session_ind, test_trial,
                #                              float(mean_error_bins))

                        # plot_decoded_bins(estimated_bins, test_bins,
                        #                   session_details)
                        #
                        # mean_error_all_sessions \
                        #     [np.abs(train_session_ind - test_session_ind)]. \
                        #     append(mean_error_bins)
                        # pval_for_mean_error \
                        #     [np.abs(train_session_ind - test_session_ind)]. \
                        #     append(pval)
                        # mean_error_permutaion_all_sessions\
                        # [np.abs(train_session_ind - test_session_ind)].\
                        #     append(mae_permutation)

        # print 'saving: linear_track_decoding_results_c%sm%s' % (CAGE[i], mouse)
        # np.savez('linear_track_decoding_results_c%sm%s' % (CAGE[i], mouse),
        #          mean_error_all_sessions=mean_error_all_sessions,
        #          mean_error_permutaion_all_sessions =
        #          mean_error_permutaion_all_sessions,
        #          pval_for_mean_error=pval_for_mean_error)

        # print 'saving: Lshape_track_decoding_results_c%sm%s' % (CAGE[i], mouse)
        # np.savez('Lshape_track_decoding_results_c%sm%s' % (CAGE[i], mouse),
        #          mean_error_all_sessions=mean_error_all_sessions,
        #          mean_error_permutaion_all_sessions=
        #          mean_error_permutaion_all_sessions,
        #          pval_for_mean_error=pval_for_mean_error)
        raw_input('press enter')
        close("all")

if __name__ == '__main__':
    main()