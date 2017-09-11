import numpy as np
import scipy.stats as sio
from matplotlib.pyplot import *
from matplotlib import cm
from decode_bucket_trials import MOUSE, CAGE, VELOCITY_THRESHOLD
from plot_SCE_analysis import divide_to_boxes

def add_subplot_axes(ax,rect,axisbg='w'):
    # Taken from: https://stackoverflow.com/questions/17458580/embedding-small
    # -plots-inside-subplots-in-matplotlib
    fig = gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def main():
    f1 = figure()
    ##### Plot decoding statistics ######
    correct_decoding_A = []
    correct_decoding_B = []
    for i, mouse in enumerate(MOUSE):
        npzfile = np.load(r'results\bucket_decoding_results_c%sm%s.npz'\
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
        if p >= 0.1:
            displaystring = r'n.s.'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'

        text(i+1, max_value + 0.05, displaystring, ha='center',
                 va='center', bbox=dict(facecolor='1.', edgecolor='none'))

    xticks([1, 2], ['A', 'B'], fontsize=22)
    yticks(fontsize=22)
    ylim(0,1)
    ylabel('Matched environment decoding fraction', fontsize=20)
    legend(handles=[line1], fontsize=20)

    f1.show()


    ###### Plot decoding histogram and number of events per bin decoding ######

    f2, axx2 = subplots(2, 2, sharex=True, sharey='row')
    decoded_bins_all_mice = {'envA': [], 'envB': []}
    decoded_env_all_mice = {'envA': [], 'envB': []}
    number_of_events_per_frame_all_mice = {'envA': [], 'envB': []}
    for i, mouse in enumerate(MOUSE):

        npzfile = np.load(r'bucket_decoding_results_c%sm%s.npz'\
                          %(CAGE[i], mouse))
        decoded_bins_all_sessions = npzfile['decoded_bins_all_sessions'].all()
        decoded_env_all_sessions = npzfile['decoded_env_all_sessions'].all()
        number_of_events_per_frame_all_sessions = npzfile \
        ['number_of_events_per_frame_all_sessions'].all()

        decoded_bins_all_mice['envA'].append\
            (np.concatenate(decoded_bins_all_sessions['envA']))
        decoded_env_all_mice['envA'].append\
            (np.concatenate(decoded_env_all_sessions['envA']))
        number_of_events_per_frame_all_mice['envA'].append\
            (np.concatenate(number_of_events_per_frame_all_sessions['envA']))
        decoded_bins_all_mice['envB'].append \
            (np.concatenate(decoded_bins_all_sessions['envB']))
        decoded_env_all_mice['envB'].append \
            (np.concatenate(decoded_env_all_sessions['envB']))
        number_of_events_per_frame_all_mice['envB'].append \
            (np.concatenate(number_of_events_per_frame_all_sessions['envB']))

    decoded_bins_all_mice['envA'] = np.concatenate(
        decoded_bins_all_mice['envA'])
    decoded_env_all_mice['envA'] = np.concatenate(decoded_env_all_mice['envA'])
    number_of_events_per_frame_all_mice['envA'] = np.concatenate(
        number_of_events_per_frame_all_mice['envA'])

    decoded_bins_all_mice['envB'] = np.concatenate(
        decoded_bins_all_mice['envB'])
    decoded_env_all_mice['envB'] = np.concatenate(decoded_env_all_mice['envB'])
    number_of_events_per_frame_all_mice['envB'] = np.concatenate(
        number_of_events_per_frame_all_mice['envB'])

    env_A_indices = {}
    env_B_indices = {}
    env_A_indices['envA'] = decoded_env_all_mice['envA'] == 0
    env_A_indices['envB'] = decoded_env_all_mice['envB'] == 0
    env_B_indices['envA'] = decoded_env_all_mice['envA'] == 1
    env_B_indices['envB'] = decoded_env_all_mice['envB'] == 1

    box_data_A = {}
    box_data_B = {}
    box_data_A['envA'] = divide_to_boxes\
        (number_of_events_per_frame_all_mice['envA'][env_A_indices['envA']],
         decoded_bins_all_mice['envA'][env_A_indices['envA']])
    box_data_A['envB'] = divide_to_boxes \
        (number_of_events_per_frame_all_mice['envB'][env_A_indices['envB']],
         decoded_bins_all_mice['envB'][env_A_indices['envB']])

    box_data_B['envA'] = divide_to_boxes \
        (number_of_events_per_frame_all_mice['envA'][env_B_indices['envA']],
         decoded_bins_all_mice['envA'][env_B_indices['envA']])
    box_data_B['envB'] = divide_to_boxes \
        (number_of_events_per_frame_all_mice['envB'][env_B_indices['envB']],
         decoded_bins_all_mice['envB'][env_B_indices['envB']])

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

    dA, pA = sio.ks_2samp(decoded_bins_all_mice['envA'][env_A_indices['envA']],
                        envA_bins[np.abs(envA_velocity) > VELOCITY_THRESHOLD])
    dB, pB = sio.ks_2samp(decoded_bins_all_mice['envB'][env_A_indices['envB']],
                        envB_bins[np.abs(envB_velocity) > VELOCITY_THRESHOLD])

    axx2[1, 0].hist([decoded_bins_all_mice['envA'][env_A_indices['envA']],
                     decoded_bins_all_mice['envB'][env_A_indices['envB']],
                     envA_bins[np.abs(envA_velocity) > VELOCITY_THRESHOLD]],
                    normed=True, align='right', bins=12)
    axx2[1, 0].set_xlabel('Decoded bin', fontsize=25)
    axx2[1, 0].set_ylabel('Probability', fontsize=25)
    axx2[1, 1].hist([decoded_bins_all_mice['envB'][env_B_indices['envB']],
                     decoded_bins_all_mice['envA'][env_B_indices['envA']],
                     envB_bins[np.abs(envB_velocity) > VELOCITY_THRESHOLD]],
                    normed=True, bins=12, align='right', label=
                    ['decoded bins in matched bucket',
                     'decoded bins in not matched bucket',
                     'linear track occupancy'])
    legend(bbox_to_anchor=(1.3, 1.2),fontsize=20)
    axx2[0, 0].boxplot(box_data_A['envA'])
    rect = [0.1, 0.5, 0.4, 0.4]
    insetA = add_subplot_axes(axx2[0, 0],rect)
    insetA.boxplot(box_data_A['envB'])
    axx2[0, 0].set_ylim(0,35)
    axx2[0, 0].set_ylabel('Number of events in frame', fontsize=25)
    axx2[0, 0].set_title('Environment A', fontsize=25)
    axx2[0, 1].boxplot(box_data_B['envB'])
    insetB = add_subplot_axes(axx2[0, 1], rect)
    insetB.boxplot(box_data_B['envA'])
    axx2[0, 1].set_title('Environment B', fontsize=25)

    axx2[1, 1].set_xlabel('Decoded bin', fontsize=25)
    f2.suptitle('Bucket decoding', fontsize=25)
    for i in range(2):
        for j in range(2):
            for xtick in axx2[i, j].xaxis.get_major_ticks():
               xtick.label.set_fontsize(22)
            for ytick in axx2[i, j].yaxis.get_major_ticks():
                ytick.label.set_fontsize(22)

    f2.show()

    ############ Plot bin representation ###########
    f2, axx2 = subplots(3, 2, sharex='row', sharey='row')
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

        f0, axx0 = subplots(3, 2, sharex='row', sharey='row')
        # axx0[0, 0].hist(np.argmax(p_neuron_bin['envB_negative'][0], axis=1))
        # axx0[0, 0].set_title('L-shape negative speed')
        # axx0[0, 0].set_ylabel('# cells with place field', fontsize=22)
        # axx0[0, 1].hist(np.argmax(p_neuron_bin['envA_negative'][0], axis=1))
        # axx0[0, 1].set_title('Linear negative speed')
        # axx0[1, 0].hist(np.argmax(p_neuron_bin['envB_positive'][0], axis=1))
        # axx0[1, 0].set_title('L-shape positive speed')
        # axx0[1, 0].set_ylabel('# cells with place field', fontsize=22)
        # axx0[1, 1].hist(np.argmax(p_neuron_bin['envA_positive'][0], axis=1))
        # axx0[1, 1].set_title('Linear positive speed')
        # axx0[1, 1].set_title('L-shape negative speed')
        # f0.suptitle('c%sm%s'% (CAGE[i], mouse))
        # f0.show()

        indices = np.argsort(
            np.argmax(p_neuron_bin['envB_negative'][0], axis=1))
        cax = axx0[0, 0].imshow(p_neuron_bin['envB_negative'][0][indices, :],
                          interpolation='none', aspect='auto', cmap=cm.viridis)
        axx0[0, 0].set_title('L-shape negative speed')
        indices = np.argsort(
            np.argmax(p_neuron_bin['envA_negative'][0], axis=1))
        axx0[0, 1].imshow(p_neuron_bin['envA_negative'][0][indices, :],
                          interpolation='none', aspect='auto', cmap=cm.viridis)
        axx0[0, 1].set_title('Linear negative speed')
        indices = np.argsort(
            np.argmax(p_neuron_bin['envB_positive'][0], axis=1))
        axx0[1, 0].imshow(p_neuron_bin['envB_positive'][0][indices, :],
                          interpolation='none', aspect='auto', cmap=cm.viridis)
        axx0[1, 0].set_title('L-shape positive speed')
        indices = np.argsort(
            np.argmax(p_neuron_bin['envA_positive'][0], axis=1))
        axx0[1, 1].imshow(p_neuron_bin['envA_positive'][0][indices, :],
                          interpolation='none', aspect='auto', cmap=cm.viridis)
        f0.colorbar(cax)
        axx0[1, 1].set_title('Linear positive speed')
        f0.suptitle('P(neuron activation|bin) C%sM%s' % (CAGE[i], mouse))
        f0.show()

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
    axx2[0, 0].set_ylabel('# cells with place field', fontsize=25)
    axx2[0, 1].hist(np.argmax(p_neuron_bin_all['envA_negative'], axis=1))
    axx2[0, 1].set_title('Linear negative speed')
    axx2[1, 0].hist(np.argmax(p_neuron_bin_all['envB_positive'], axis=1))
    axx2[1, 0].set_title('L-shape positive speed')
    axx2[1, 0].set_ylabel('# cells with place field', fontsize=25)
    axx2[1, 1].hist(np.argmax(p_neuron_bin_all['envA_positive'], axis=1))
    axx2[1, 1].set_title('Linear positive speed')
    axx2[1, 1].set_title('L-shape negative speed')
    f2.suptitle('P(neuron activation|bin) all mice')

    f2.show()

    # envA_bins = []
    # envB_bins = []
    # envA_velocity = []
    # envB_velocity = []
    #
    # for i, mouse in enumerate(MOUSE):
    #     npzfile = np.load(r'results\bins_velocity_c%sm%s.npz' \
    #                       % (CAGE[i], mouse))
    #     bins = npzfile['all_bins'].all()
    #     velocity = npzfile['all_velocity'].all()
    #     envA_bins.append(np.concatenate(bins['envA']))
    #     envB_bins.append(np.concatenate(bins['envB']))
    #
    #     envA_velocity.append(np.concatenate(velocity['envA']))
    #     envB_velocity.append(np.concatenate(velocity['envB']))
    #
    # envA_bins = np.concatenate(envA_bins)
    # envB_bins = np.concatenate(envB_bins)
    # envA_velocity = np.concatenate(envA_velocity)
    # envB_velocity = np.concatenate(envB_velocity)
    # box_dataA = divide_to_boxes(np.abs(envA_velocity), envA_bins)
    # box_dataB = divide_to_boxes(np.abs(envB_velocity), envB_bins)
    #
    # axx2[2, 1].boxplot(box_dataA)
    # axx2[2, 1].set_title('Linear track')
    # axx2[2, 1].set_xlabel('# bin', fontsize=25)
    # axx2[2, 0].boxplot(box_dataB)
    # axx2[2, 0].set_title('L-shape track')
    # axx2[2, 0].set_ylim(0,100)
    # axx2[2, 0].set_xlabel('# bin', fontsize=25)
    # axx2[2, 0].set_ylabel('speed (cm/sec)', fontsize=25)
    #
    # f2.suptitle('Speed distribution per bin', fontsize=25)
    #
    # f2.show()

    raw_input('press enter')

if __name__ == '__main__':
    main()