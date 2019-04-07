import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict

from decode_bucket_trials import load_session_data
from bambi.tools import matlab, activity_loading
import synchronous_calcium_events as sce


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
NUMBER_OF_PERMUTATIONS = 1000

# Create colormap for figure
grays = cm.get_cmap('gray_r', 256)
newcolors = grays(np.linspace(0, 1, 256))
red = np.array([1, 0, 0, 1])
newcolors[-25:, :] = red
COLORMAP = ListedColormap(newcolors)

def create_sce_population_vector(events, sce_mask, window_size, frame_rate):
    sce_frame_size = int(window_size*frame_rate)
    sce_population_vectors = []
    sce_locations = np.reshape(np.argwhere(sce_mask), (1, -1))[0]
    for loc in sce_locations:
        sce_events = events[:, loc:loc+sce_frame_size]
        sce_population_vectors.append(np.sum(sce_events, axis=1)/
                                      float(sce_frame_size))

    return sce_population_vectors, sce_locations


def create_shuffled_sce(event_rates, number_of_active_neurons,
                        number_of_permutations):
    number_of_neurons = event_rates.shape[0]
    shuffled_sces = np.zeros((number_of_neurons, number_of_permutations))
    event_rates_cdf = np.cumsum(event_rates)
    event_rates_cdf = np.concatenate([[0], event_rates_cdf])
    for i in range(number_of_permutations):
        rand_indices = np.random.uniform(0, event_rates_cdf[-1],
                                         size=int(number_of_active_neurons))
        neuron_indices = np.digitize(rand_indices, event_rates_cdf) - 1
        shuffled_sces[neuron_indices, i] = 1

    return shuffled_sces


def find_sce_correlation_significance_level(events, number_of_active_neurons,
                                            trial_population_vector,
                                            number_of_permutations, alpha=0.05):
    # Find the unique numbers of number of neurons in SCE
    unique_number_of_active_neurons = np.unique(number_of_active_neurons)

    # Create a distribution of event rates
    number_of_frames = events.shape[1]
    event_rate_distribution = np.sum(events, axis=1)/float(number_of_frames)

    # For each number of neurons in SCE, find the significance level by creating
    # random vectors of activity with the same number of active neurons and
    # correlating this vectors with the trial's activity
    significance_correlation_level = \
        np.zeros_like(unique_number_of_active_neurons)

    for i, events_number in enumerate(unique_number_of_active_neurons):
        shuffled_sces = create_shuffled_sce(
            event_rate_distribution, events_number, number_of_permutations)
        shuffled_correlations = [np.corrcoef(x, trial_population_vector)[0, 1]
                                 for x in shuffled_sces.T]
        correlations_distribution, edge_bins = np.histogram(
            shuffled_correlations, bins=100, density=True)
        significance_correlation_level[i] = \
            edge_bins[np.argwhere(np.cumsum(correlations_distribution*(
                edge_bins[1] - edge_bins[0])) > 1-alpha)[0][0]]

    return significance_correlation_level, unique_number_of_active_neurons

    
def plot_sce_activity_for_all_trial(events, sce_chance_level, sce_locations,
                                     significant_sce_correlation):
    number_of_active_neurons = sce.count_events_in_sliding_window(
        events, FRAME_RATE)
    max_active_neurons = np.max(number_of_active_neurons)
    number_of_significant_sces = np.sum(significant_sce_correlation)
    number_of_frames = len(number_of_active_neurons)
    t = np.arange(number_of_frames) / float(FRAME_RATE)
    plt.figure()
    plt.plot(t, number_of_active_neurons, 'b')
    plt.plot(t, sce_chance_level*np.ones(number_of_frames), 'r')
    plt.plot(t[sce_locations[significant_sce_correlation]], 
            (max_active_neurons + 0.2)*np.ones(number_of_significant_sces), '*k')
    plt.xlabel('Time [sec]', fontsize=16)
    plt.ylabel('Number of active neurons', fontsize=16)
    plt.show()

    return


def plot_raster_of_the_SCE_activity(bucket_events, sce_locations,
                                    significant_sce_mask):
    number_of_frames = bucket_events.shape[1]
    # Fill the locations of SCE occurrence
    full_sce_locations = []
    for j in range(int(sce.WINDOW * FRAME_RATE)):
        full_sce_locations.append(sce_locations + j)
    full_sce_locations = np.concatenate(full_sce_locations)
    full_sce_locations = np.sort(full_sce_locations)

    significant_locations = sce_locations[significant_sce_mask]
    significant_sce_locations = []
    for j in range(int(sce.WINDOW * FRAME_RATE)):
        significant_sce_locations.append(significant_locations + j)
    significant_sce_locations = np.concatenate(significant_sce_locations)
    significant_sce_locations = np.sort(significant_sce_locations)

    # Find the neurons that are active only at SCE
    active_neurons_in_sce = np.argwhere(np.sum(
        bucket_events[:, full_sce_locations], axis=1) > 0)[:, 0]
    sce_bucket_events = np.zeros_like(bucket_events, dtype=int)
    sce_bucket_events[:, full_sce_locations] = bucket_events[
                                               :, full_sce_locations]
    sce_bucket_events = sce_bucket_events[active_neurons_in_sce, :]

    significant_sce_events = np.zeros_like(bucket_events, dtype=int)
    significant_sce_events[:, significant_sce_locations] = bucket_events[
                                               :, significant_sce_locations]
    significant_sce_events = significant_sce_events[active_neurons_in_sce, :]

    bucket_events = bucket_events[active_neurons_in_sce, :]

    # Sort the neurons
    first_activation_time = np.argmax(sce_bucket_events, axis=1)
    ind_sort = np.argsort(first_activation_time)

    sce_bucket_events = sce_bucket_events[ind_sort, :]
    significant_sce_events = significant_sce_events[ind_sort, :]
    bucket_events = bucket_events[ind_sort, :]

    figure_events = bucket_events + 3 * sce_bucket_events + \
        7 * significant_sce_events

    plt.matshow(figure_events, aspect='auto',
                interpolation='none', cmap=COLORMAP)
    plt.xticks(np.arange(0, number_of_frames, 100),
               np.arange(0, number_of_frames/float(FRAME_RATE),
                                100/float(FRAME_RATE)), fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().xaxis.tick_bottom()
    plt.xlabel('Time [sec]', fontsize=20)
    plt.ylabel('# Neuron', fontsize=20)
    plt.show()
    return


def plot_cell_activity(axx, cell_events, bins, velocity=[]):
    number_of_frames = len(bins)
    only_events = np.zeros(number_of_frames)
    t = np.arange(number_of_frames) / float(FRAME_RATE)
    only_events[cell_events == False] = np.nan
    only_events[cell_events == True] = 1
    events_color = np.random.rand(3)
    axx.plot(t, bins, 'gray')
    axx.plot(t, only_events*bins, '*', color=events_color, markersize=10)
    max_bin = np.max(bins)
    fig = axx.set_ylim([-0.1, max_bin+0.1])
    if len(velocity) > 0:
        axx.plot(t, velocity, 'g')
        max_velocity = np.max(velocity)
        min_velocity = np.min(velocity)
        fig = axx.set_ylim([-1 + min_velocity, 1 + max_velocity])
    return fig


def plot_cells_activity_on_track(cells_activity, bins, velocity=[]):
    number_of_cells = cells_activity.shape[0]
    f, axx = plt.subplots(number_of_cells, 1)
    for cell in range(number_of_cells):
        plot_cell_activity(axx[cell], cells_activity[cell, :], bins,
                           velocity=velocity)
    f.show()
    return


def plot_all_cells_activity_on_track(cells_activity, bins, velocity=[]):
    number_of_cells = cells_activity.shape[0]
    f, axx = plt.subplots(1, 1)
    for cell in range(number_of_cells):
        plot_cell_activity(axx, cells_activity[cell, :], bins,
                           velocity=velocity)
    axx.set_xlabel('Time [sec]', fontsize=16)
    axx.set_ylabel('Location bin', fontsize=16)
    
    f.show()
    return


def main():
    for i, mouse in enumerate(MOUSE):
        cell_reg_filename = WORK_DIR + '\c%dm%d\cellRegisteredFixed.mat' \
                                       %(CAGE[i], mouse)
        cell_registration = matlab.load_cell_registration(cell_reg_filename)

        for session_ind, day in enumerate(DAYS):
            print 'C%dM%d day %s' % (CAGE[i], mouse, day)
            # Create training data for environment A
            session_dir = WORK_DIR + '\c%dm%d\day%s\%s' \
                                     % (CAGE[i], mouse, day, ENV[0])
            events_tracesA, movement_dataA = load_session_data \
                (session_dir, cell_registration, 2*session_ind)
            linear_trials_indicesA = range(len(events_tracesA))[1:-1]
            bucket_trials_indicesA = [0, len(events_tracesA) - 1]

            # Create training data from one linear track trial
            linear_training_trial = 5
            [train_bins, train_events] = activity_loading.create_training_data \
                (movement_dataA, events_tracesA, [linear_training_trial])
            train_events = train_events > 0
            train_bins = activity_loading.wide_binning(train_bins, NUMBER_OF_BINS,
                                                  SPACE_BINNING)
            train_velocity = activity_loading.concatenate_movment_data(
                movement_dataA, 'velocity', [linear_training_trial])

            # Calculate the population vector of the trial:
            positive_velocity_indices = train_velocity > VELOCITY_THRESHOLD
            negative_velocity_indices = train_velocity < VELOCITY_THRESHOLD
            positive_trial_population_vector = np.sum(
                train_events[:, positive_velocity_indices], axis=1) / \
                                       float(len(positive_velocity_indices))
            negative_trial_population_vector = np.sum(
                train_events[:, negative_velocity_indices], axis=1) / \
                                           float(len(negative_velocity_indices))

            # Calculate SCEs in bucket trials
            bucket_trial_index = 1
            bucket_events = events_tracesA[bucket_trial_index] > 0
            sce_chance_level = sce.calculte_SCE_chance_level(
                bucket_events, FRAME_RATE)
            print sce_chance_level
            sce_mask = sce.find_SCE_in_full_epoch(bucket_events,
                                                  sce_chance_level, FRAME_RATE)
            sce_population_vectors, sce_locations = create_sce_population_vector(
                bucket_events, sce_mask, sce.WINDOW, FRAME_RATE)

            # Correlate between the SCE population vectors to the trial's
            # population vector
            number_of_sces = len(sce_population_vectors)
            sce_activity_corr_positive = np.zeros(number_of_sces)
            sce_activity_corr_negative = np.zeros(number_of_sces)
            number_of_active_neurons = np.zeros(number_of_sces)
            for j, sce_pv in enumerate(sce_population_vectors):
                number_of_active_neurons[j] = np.sum(sce_pv > 0)
                sce_activity_corr_positive[j] = np.corrcoef(
                    sce_pv, positive_trial_population_vector)[0, 1]
                sce_activity_corr_negative[j] = np.corrcoef(
                    sce_pv, negative_trial_population_vector)[0, 1]


            # Caclculate the significance of results of positive direction
            correlation_significance_level, unique_number_of_active_neurons = \
                find_sce_correlation_significance_level(
                    bucket_events, number_of_active_neurons,
                    positive_trial_population_vector, NUMBER_OF_PERMUTATIONS,
                    alpha=0.025)

            positive_significant_sce_correlation = np.zeros_like(sce_activity_corr_positive,
                                                        dtype=bool)
            for j, sce_corr in enumerate(sce_activity_corr_positive):
                n = number_of_active_neurons[j]
                significance_level = correlation_significance_level[np.argwhere(
                    unique_number_of_active_neurons == n)[0][0]]
                positive_significant_sce_correlation[j] = sce_corr >= significance_level
            
          
            print '%d out of %d significant SCEs (positive direction)' \
                % (np.sum(positive_significant_sce_correlation), number_of_sces)
            

            # Caclculate the significance of results of negative direction
            correlation_significance_level, unique_number_of_active_neurons = \
                find_sce_correlation_significance_level(
                    bucket_events, number_of_active_neurons,
                    negative_trial_population_vector, NUMBER_OF_PERMUTATIONS,
                    alpha=0.025)

            negative_significant_sce_correlation = np.zeros_like(
                sce_activity_corr_negative,
                dtype=bool)
            for j, sce_corr in enumerate(sce_activity_corr_negative):
                n = number_of_active_neurons[j]
                significance_level = correlation_significance_level[np.argwhere(
                    unique_number_of_active_neurons == n)[0][0]]
                negative_significant_sce_correlation[j] = sce_corr >= significance_level

            print '%d out of %d significant SCEs (negative direction)' \
                  % (np.sum(negative_significant_sce_correlation), number_of_sces)
            
            all_significant_sce_correlation = \
                positive_significant_sce_correlation | \
                negative_significant_sce_correlation
            
            # Plot raster of the SCE activity
            plot_raster_of_the_SCE_activity(bucket_events, sce_locations,
                                            all_significant_sce_correlation)
            
            # Plot SCE's cell activity on track - each cell separatly
            # significant_sce_locations = sce_locations[all_significant_sce_correlation]
            # for s in significant_sce_locations:
                # sce_activity = bucket_events[:, s:s+int(sce.WINDOW*FRAME_RATE)]
                # active_neurons = np.sum(sce_activity, axis=1) > 0
                # active_neurons_events = train_events[active_neurons, :]
                # plot_cells_activity_on_track(active_neurons_events, train_bins)
                # raw_input('enter')
            
            # Plot SCE's cell activity on track - together
            significant_sce_locations = sce_locations[all_significant_sce_correlation]
            for s in significant_sce_locations:
                sce_activity = bucket_events[:, s:s+int(sce.WINDOW*FRAME_RATE)]
                active_neurons = np.sum(sce_activity, axis=1) > 0
                active_neurons_events = train_events[active_neurons, :]
                plot_all_cells_activity_on_track(active_neurons_events, train_bins)
                raw_input('enter')


if __name__ == '__main__':
    main()