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
            if environment_name=='envA':
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

def main():
    for i, mouse in enumerate(MOUSE):
        cell_reg_filename = WORK_DIR + '\c%dm%d\cellRegisteredFixed.mat' %(CAGE[i], mouse)
        cell_registration = matlab.load_cell_registration(cell_reg_filename)
        correct_decoding_percentage = []
        edge_decoding_percentage = []
        for session_ind, day in enumerate(DAYS):
            p_neuron_bin = {}
            # Create p_neuron_bin for envA
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' %(CAGE[i], mouse, day, ENV[0])
            events_tracesA, movement_dataA = load_session_data(session_dir, cell_registration, 2*session_ind)
            linear_trials_indicesA = range(len(events_tracesA))[1:-1]
            bucket_trials_indicesA = [0, len(events_tracesA) - 1]
            [bins, events] = create_training_data(movement_dataA, events_tracesA, linear_trials_indicesA)
            bins = wide_binning(bins, 24, 2)
            p_neuron_binA = maximum_likelihood.calculate_p_r_s_matrix(bins, events)
            p_neuron_bin['envA'] = p_neuron_binA

            # Create p_neuron_bin for envB
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' % (CAGE[i], mouse, day, ENV[1])
            events_tracesB, movement_dataB = load_session_data(session_dir, cell_registration, 2*session_ind+1)
            linear_trials_indicesB = range(len(events_tracesB))[1:-1]
            bucket_trials_indicesB = [0, len(events_tracesB)-1]
            [bins, events] = create_training_data(movement_dataB, events_tracesB, linear_trials_indicesB)
            bins = wide_binning(bins, 24, 2)
            p_neuron_binB = maximum_likelihood.calculate_p_r_s_matrix(bins, events)
            p_neuron_bin['envB'] = p_neuron_binB

            for trial in range(2):
                trial_events_A = events_tracesA[bucket_trials_indicesA[trial]]
                statistics, _, _ = test_bucket_trial(trial_events_A, p_neuron_bin, EDGE_BINS)
                correct_decoding_percentage.append(statistics['envA']['overall_decoding_fraction'])
                edge_decoding_percentage.append(statistics['envA']['edge_decoding_fraction'])

                trial_events_B = events_tracesB[bucket_trials_indicesB[trial]]
                statistics, _, _ = test_bucket_trial(trial_events_B, p_neuron_bin, EDGE_BINS)
                correct_decoding_percentage.append(statistics['envB']['overall_decoding_fraction'])
                edge_decoding_percentage.append(statistics['envB']['edge_decoding_fraction'])

        f, axx = subplots(2, 1, sharex=True)
        env_label = ['A', 'B']*14
        axx[0].bar(range(4*7), correct_decoding_percentage, tick_label=env_label)
        axx[0].set_title('correct decoding fraction c%sm%s' %(CAGE[i], mouse))
        axx[0].set_ylabel('fraction')
        axx[0].set_ylim((0,1.1))

        axx[1].bar(range(4*7), edge_decoding_percentage, tick_label=env_label)
        axx[1].set_title('edge decoding fraction c%sm%s' % (CAGE[i], mouse))
        axx[1].set_ylabel('fraction')
        f.show()

    raw_input('press enter')

main()






