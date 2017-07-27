import numpy as np
import scipy.stats as sio
from matplotlib.pyplot import *
from decode_bucket_trials import MOUSE, CAGE, VELOCITY_THRESHOLD
from plot_SCE_analysis import divide_to_boxes

def main():
    f1 = figure()
    ##### Plot decoding statistics ######
    correct_decoding_A = []
    correct_decoding_B = []
    for i, mouse in enumerate(MOUSE):
        npzfile = np.load(r'results\bucket_decoding_statistics_c%sm%s.npz'\
                          %(CAGE[i], mouse))
        correct_decoding_percentage = npzfile['correct_decoding_percentage']

        correct_decoding_A.append(correct_decoding_percentage[0:28:4])
        correct_decoding_A.append(correct_decoding_percentage[2:28:4])
        correct_decoding_B.append(correct_decoding_percentage[1:28:4])
        correct_decoding_B.append(correct_decoding_percentage[3:28:4])

    correct_decoding_A = np.concatenate(correct_decoding_A)
    correct_decoding_B = np.concatenate(correct_decoding_B)

    tA, probA = sio.ttest_1samp(correct_decoding_A, 0.5)
    tB, probB = sio.ttest_1samp(correct_decoding_B, 0.5)

    boxplot([correct_decoding_A, correct_decoding_B])
    line1, = plot(np.arange(0, 3.5, 0.5), np.ones(7)*0.5, '--r',
        label='Chance level')
    # plot significance star inspired from:
    # https://stackoverflow.com/questions/33873176/how-to-add-significance-levels-on-bar-graph-using-pythons-matplotlib

    pvals = [probA, probB]
    max_value = np.max(np.concatenate([correct_decoding_A, correct_decoding_B]))
    for i, p in enumerate(pvals):
        if p >= 0.025:
            displaystring = r'n.s.'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'

        text(i+1, max_value + 0.1, displaystring, ha='center',
                 va='center', bbox=dict(facecolor='1.', edgecolor='none'))

    xticks([1, 2], ['A', 'B'], fontsize=15)
    yticks(fontsize=15)
    ylim(0,1)
    ylabel('Matched environment decoding fraction', fontsize=15)
    legend(handles=[line1])

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

    dA, pA = sio.ks_2samp(decoded_bins_all_mice[env_A_indices],
                        envA_bins[np.abs(envA_velocity) > VELOCITY_THRESHOLD])
    dB, pB = sio.ks_2samp(decoded_bins_all_mice[env_A_indices],
                        envA_bins[np.abs(envA_velocity) > VELOCITY_THRESHOLD])
    axx2[0, 0].boxplot(box_data_A)
    axx2[0, 0].set_ylim(0,35)
    axx2[0, 0].set_ylabel('Number of events in frame', fontsize=18)
    axx2[0, 0].set_title('Environment A', fontsize=18)
    axx2[0, 1].boxplot(box_data_B)
    axx2[0, 1].set_title('Environment B', fontsize=18)
    axx2[1, 0].hist([decoded_bins_all_mice[env_A_indices],
                     envA_bins[np.abs(envA_velocity) > VELOCITY_THRESHOLD]],
                    normed=True, align='right')
    axx2[1, 0].set_xlabel('Decoded bin', fontsize=18)
    axx2[1, 0].set_ylabel('Probability', fontsize=18)
    axx2[1, 1].hist([decoded_bins_all_mice[env_B_indices],
                     envB_bins[np.abs(envB_velocity) > VELOCITY_THRESHOLD]],
                    normed=True, align='right',label=
                    ['decoded bins distribution','linear track occupancy'])
    axx2[1, 1].set_xlabel('Decoded bin', fontsize=18)
    f2.suptitle('Bucket decoding', fontsize=18)
    legend()
    for i in range(2):
        for j in range(2):
            for xtick, ytick in zip(axx2[i, j].xaxis.get_major_ticks(),
                                    axx2[i, j].yaxis.get_major_ticks()):
               xtick.label.set_fontsize(15)
               ytick.label.set_fontsize(15)

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