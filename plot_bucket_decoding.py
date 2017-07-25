import numpy as np
from matplotlib.pyplot import *
from decode_bucket_trials import MOUSE, CAGE
from plot_SCE_analysis import divide_to_boxes


def main():
    f1, axx1 = subplots(1, 2, sharey=True, sharex=True)
    ##### Plot decoding statistics ######
    for i, mouse in enumerate(MOUSE):
        npzfile = np.load(r'results\bucket_decoding_statistics_c%sm%s.npz'\
                          %(CAGE[i], mouse))
        correct_decoding_percentage = npzfile['correct_decoding_percentage']
        p_val_correct = npzfile['p_val_correct']
        edge_decoding_percentage = npzfile['edge_decoding_percentage']
        p_val_edge = npzfile['p_val_edge']

        # plot all bucket trials env A and B
        axx1[0].plot(p_val_correct[0:28:4], correct_decoding_percentage[0:28:4],
                     'ro') #A
        axx1[0].plot(p_val_correct[1:28:4], correct_decoding_percentage[1:28:4],
                     'bo') #B
        axx1[0].plot(p_val_correct[2:28:4], correct_decoding_percentage[2:28:4],
                     'ro') #A
        axx1[0].plot(p_val_correct[3:28:4], correct_decoding_percentage[3:28:4],
                     'bo') #B

        axx1[1].plot(p_val_edge[0:28:4], edge_decoding_percentage[0:28:4],
                     'ro')
        axx1[1].plot(p_val_edge[1:28:4], edge_decoding_percentage[1:28:4],
                     'bo')
        axx1[1].plot(p_val_edge[2:28:4], edge_decoding_percentage[2:28:4],
                     'ro')
        axx1[1].plot(p_val_edge[3:28:4], edge_decoding_percentage[3:28:4],
                     'bo')

    axx1[0].set_ylim((-0.1, 1.1))
    axx1[0].set_xlim((-0.1, 1.1))
    axx1[0].set_title('P value Vs. correct decoding fraction', fontsize=18)
    axx1[0].set_ylabel('correct decoding fraction', fontsize=16)
    axx1[0].set_xlabel('P value', fontsize=16)
    axx1[1].set_ylim((-0.1, 1.1))
    axx1[1].set_xlim((-0.1, 1.1))
    axx1[1].set_title('P value Vs. edge decoding fraction', fontsize=18)
    axx1[1].set_ylabel('edge decoding fraction', fontsize=16)
    axx1[1].set_xlabel('P value', fontsize=16)

    for i in range(2):
        for xtick, ytick in zip(axx1[i].xaxis.get_major_ticks(),
                                axx1[i].yaxis.get_major_ticks()):
            xtick.label.set_fontsize(15)
            ytick.label.set_fontsize(15)


    f1.show()

    ###### Plot decoding histogram and number of events per bin decoding ######

    f2, axx2 = subplots(2, 2, sharex=True, sharey='row')
    decoded_bins_all_mice = []
    decoded_env_all_mice = []
    number_of_events_per_frame_all_mice = []
    for i, mouse in enumerate(MOUSE):

        npzfile = np.load(r'results\bucket_decoding_results_c%sm%s.npz'\
                          %(CAGE[i], mouse))
        decoded_bins_all_sessions = npzfile['decoded_bins_all_sessions']
        decoded_env_all_sessions = npzfile['decoded_env_all_sessions']
        number_of_events_per_frame_all_sessions = npzfile \
        ['number_of_events_per_frame_all_sessions']

        decoded_bins_all_mice.append(np.concatenate(decoded_bins_all_sessions))
        decoded_env_all_mice.append(np.concatenate(decoded_env_all_sessions))
        number_of_events_per_frame_all_mice.append(np.concatenate(
            number_of_events_per_frame_all_sessions))

    decoded_bins_all_mice = np.concatenate(decoded_bins_all_mice)
    decoded_env_all_mice = np.concatenate(decoded_env_all_mice)
    number_of_events_per_frame_all_mice = np.concatenate(
        number_of_events_per_frame_all_mice)

    env_A_indices = decoded_env_all_mice == 0
    env_B_indices = decoded_env_all_mice == 1

    box_data_A = divide_to_boxes\
        (number_of_events_per_frame_all_mice[env_A_indices],
         decoded_bins_all_mice[env_A_indices])

    box_data_B = divide_to_boxes\
         (number_of_events_per_frame_all_mice[env_B_indices],
          decoded_bins_all_mice[env_B_indices])

    axx2[0, 0].boxplot(box_data_A)
    axx2[0, 0].set_ylim(0,20)
    axx2[0, 0].set_xlabel('Decoded bin')
    axx2[0, 0].set_ylabel('Number of events in frame')
    axx2[0, 0].set_title('Linear track')
    axx2[0, 1].boxplot(box_data_B)
    axx2[0, 1].set_xlabel('Decoded bin')
    axx2[0, 1].set_ylabel('Number of events in frame')
    axx2[0, 1].set_title('L-shape track')
    axx2[1, 0].hist(decoded_bins_all_mice[env_A_indices], normed=True)
    axx2[1, 0].set_xlabel('Decoded bin')
    axx2[1, 0].set_ylabel('Probability of decoding')
    axx2[1, 1].hist(decoded_bins_all_mice[env_B_indices], normed=True)
    axx2[1, 1].set_xlabel('Decoded bin')
    axx2[1, 1].set_ylabel('Probability of decoding')
    f2.suptitle('Bucket decoding', fontsize=18)
    f2.show()

    ############ Plot bin representation ###########
    f2, axx2 = subplots(3, 2, sharex=True, sharey='row')
    p_neuron_bin_all =  {'envB_negative': [],
                         'envA_negative': [],
                         'envB_positive': [],
                         'envA_positive': []}
    for i, mouse in enumerate(MOUSE):
        npzfile = np.load(r'results\p_neuron_bin_c%sm%s.npz' \
                          % (CAGE[i], mouse))
        p_neuron_bin = npzfile['p_neuron_bin'].all()
        # Plot the histogram of the place fields per bin for all neurons.
        # separating environments and directions
        p_neuron_bin_all['envB_negative'].append(
            p_neuron_bin['envB_negative'][0])
        p_neuron_bin_all['envA_negative'].append(
            p_neuron_bin['envA_negative'][0])
        p_neuron_bin_all['envB_positive'].append(
            p_neuron_bin['envB_positive'][0])
        p_neuron_bin_all['envA_positive'].append(
            p_neuron_bin['envA_positive'][0])

    p_neuron_bin_all['envB_negative'] = np.vstack(
        p_neuron_bin_all['envB_negative'])
    p_neuron_bin_all['envA_negative'] = np.vstack(
        p_neuron_bin_all['envA_negative'])
    p_neuron_bin_all['envB_positive'] = np.vstack(
        p_neuron_bin_all['envB_positive'])
    p_neuron_bin_all['envA_positive'] = np.vstack(
        p_neuron_bin_all['envA_positive'])

    axx2[0, 0].hist(np.argmax(p_neuron_bin_all['envB_negative'], axis=1))
    axx2[0, 0].set_title('L-shape negative speed')
    axx2[0, 0].set_ylabel('# cells with place field', fontsize=15)
    axx2[0, 1].hist(np.argmax(p_neuron_bin_all['envA_negative'], axis=1))
    axx2[0, 1].set_title('Linear negative speed')
    axx2[1, 0].hist(np.argmax(p_neuron_bin_all['envB_positive'], axis=1))
    axx2[1, 0].set_title('L-shape positive speed')
    axx2[1, 0].set_ylabel('# cells with place field', fontsize=15)
    axx2[1, 1].hist(np.argmax(p_neuron_bin_all['envA_positive'], axis=1))
    axx2[1, 1].set_title('Linear positive speed')
    axx2[1, 1].set_title('L-shape negative speed')
    f2.suptitle('P(neuron activation|bin) all mice')

        # indices = np.argsort(np.argmax(p_neuron_bin['envB_negative'][0], axis=1))
        # axx2[0, 0].imshow(p_neuron_bin['envB_negative'][0][indices, :],
        #                   interpolation='none', aspect='auto')
        # axx2[0, 0].set_title('L-shape negative speed')
        # indices = np.argsort(np.argmax(p_neuron_bin['envA_negative'][0], axis=1))
        # axx2[0, 1].imshow(p_neuron_bin['envA_negative'][0][indices , :],
        #                   interpolation='none', aspect='auto')
        # axx2[0, 1].set_title('Linear negative speed')
        # indices = np.argsort(np.argmax(p_neuron_bin['envB_positive'][0], axis=1))
        # axx2[1, 0].imshow(p_neuron_bin['envB_positive'][0][indices, :],
        #                   interpolation='none', aspect='auto')
        # axx2[1, 0].set_title('L-shape positive speed')
        # indices = np.argsort(np.argmax(p_neuron_bin['envA_positive'][0], axis=1))
        # axx2[1, 1].imshow(p_neuron_bin['envA_positive'][0][indices, :],
        #                   interpolation='none', aspect='auto')
        # axx2[1, 1].set_title('Linear positive speed')
        # f2.suptitle('P(neuron activation|bin) C%sM%s' % (CAGE[i], mouse))
    f2.show()

    envA_bins = []
    envB_bins = []
    envA_velocity = []
    envB_velocity = []

    for i, mouse in enumerate(MOUSE):
        npzfile = np.load(r'results\bins_velocity_c%sm%s.npz' \
                          % (CAGE[i], mouse))
        bins = npzfile['all_bins'].all()
        velocity = npzfile['all_velocity'].all()
        envA_bins.append(np.concatenate(bins['envA']))
        envB_bins.append(np.concatenate(bins['envB']))

        envA_velocity.append(np.concatenate(velocity['envA']))
        envB_velocity.append(np.concatenate(velocity['envB']))

    envA_bins = np.concatenate(envA_bins)
    envB_bins = np.concatenate(envB_bins)
    envA_velocity = np.concatenate(envA_velocity)
    envB_velocity = np.concatenate(envB_velocity)
    box_dataA = divide_to_boxes(np.abs(envA_velocity), envA_bins)
    box_dataB = divide_to_boxes(np.abs(envB_velocity), envB_bins)

    axx2[2, 1].boxplot(box_dataA)
    axx2[2, 1].set_title('Linear track')
    axx2[2, 1].set_xlabel('# bin', fontsize=15)
    axx2[2, 0].boxplot(box_dataB)
    axx2[2, 0].set_title('L-shape track')
    axx2[2, 0].set_ylim(0,100)
    axx2[2, 0].set_xlabel('# bin', fontsize=15)
    axx2[2, 0].set_ylabel('speed (cm/sec)', fontsize=15)

    f2.suptitle('Speed distribution per bin', fontsize=18)

    f2.show()

    raw_input('press enter')

if __name__ == '__main__':
    main()