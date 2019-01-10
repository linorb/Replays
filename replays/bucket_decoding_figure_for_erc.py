import numpy as np
import matplotlib.pyplot as plt

from decode_bucket_trials import load_session_data
from bambi.tools import matlab, activity_loading
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


def test_bucket_trial(test_trial, p_neuron_bin):
    number_of_frames = test_trial.shape[1]
    decoded_bins = np.zeros((number_of_frames))
    for frame in range(number_of_frames):
        trial_frame = np.reshape(test_trial[:, frame], (-1, 1))
        if np.sum(trial_frame) > 0:
            decoded_bins[frame], _ = maximum_likelihood.estimate_maximum_likelihood(
                trial_frame, [p_neuron_bin])
        else:
            decoded_bins[frame] = np.nan

    return decoded_bins


def main():
    for i, mouse in enumerate(MOUSE):
        cell_reg_filename = WORK_DIR + '\c%dm%d\cellRegisteredFixed.mat' \
                                       %(CAGE[i], mouse)
        cell_registration = matlab.load_cell_registration(cell_reg_filename)

        for session_ind, day in enumerate(DAYS):
            print CAGE[i], mouse, day
            # Create training data for environment A
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' \
                                     % (CAGE[i], mouse, day, ENV[0])
            events_tracesA, movement_dataA = load_session_data \
                (session_dir, cell_registration, 2*session_ind)
            linear_trials_indicesA = range(len(events_tracesA))[1:-1]
            bucket_trials_indicesA = [0, len(events_tracesA) - 1]

            # Create training data from one linear track trial and one bucket
            # trial - the bucket bin will be bin no. 12
            linear_training_trial = 1
            [train_bins, train_events] = activity_loading.create_training_data \
                (movement_dataA, events_tracesA, [linear_training_trial])
            train_events = train_events > 0
            train_bins = activity_loading.wide_binning(train_bins, NUMBER_OF_BINS,
                                                  SPACE_BINNING)

            bucket_training_trial = 0
            bucket_events = events_tracesA[bucket_training_trial] > 0
            bucket_bins = np.ones(bucket_events.shape[1])*12

            # use only events of place cells:
            [binsA, eventsA] = activity_loading.create_training_data \
                (movement_dataA, events_tracesA, linear_trials_indicesA)
            eventsA = eventsA > 0
            binsA = activity_loading.wide_binning(binsA, NUMBER_OF_BINS,
                                                  SPACE_BINNING)
            velocityA = activity_loading.concatenate_movment_data\
                (movement_dataA, 'velocity', linear_trials_indicesA)

            velocity_positive = velocityA > VELOCITY_THRESHOLD
            velocity_negative = velocityA < -VELOCITY_THRESHOLD
            place_cells_positive, _, _ = find_place_cells\
                (binsA[velocity_positive], eventsA[:, velocity_positive])

            place_cells_negative, _, _ = find_place_cells\
                (binsA[velocity_negative], eventsA[:, velocity_negative])

            place_cellsA = np.concatenate\
                ([place_cells_positive, place_cells_negative])

            # Concatenate the training data
            training_bins = np.concatenate([train_bins, bucket_bins])
            training_bins = np.array(training_bins, dtype=int)
            training_events = np.hstack([train_events[place_cellsA, :],
                                              bucket_events[place_cellsA, :]])

            # Calculate the probability matrix
            p_neuron_bin = maximum_likelihood.calculate_p_r_s_matrix(
                training_bins, training_events)

            # Test the second bucket trial
            test_trial = events_tracesA[bucket_trials_indicesA[1]][
                                place_cellsA, :]
            decoded_bins = test_bucket_trial(test_trial, p_neuron_bin)

            plt.figure()
            plt.plot(decoded_bins, '*')
            plt.show()

    raw_input('enter')

if __name__ == '__main__':
    main()