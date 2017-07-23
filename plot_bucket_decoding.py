import numpy as np
from matplotlib.pyplot import *
from decode_bucket_trials import MOUSE, CAGE
from plot_SCE_analysis import divide_to_boxes


def main():
    # f1, axx1 = subplots(1, 2, sharey=True, sharex=True)
    ###### Plot decoding statistics ######
    # for i, mouse in enumerate(MOUSE):
    #     npzfile = np.load('bucket_decoding_statistics_c%sm%s.npz'\
    #                       %(CAGE[i], mouse))
    #     correct_decoding_percentage = npzfile['correct_decoding_percentage']
    #     p_val_correct = npzfile['p_val_correct']
    #     edge_decoding_percentage = npzfile['edge_decoding_percentage']
    #     p_val_edge = npzfile['p_val_edge']
    #
    #     # plot all bucket trials env A and B
    #     axx1[0].plot(p_val_correct[0:28:4], correct_decoding_percentage[0:28:4],
    #                  'ro') #A
    #     axx1[0].plot(p_val_correct[1:28:4], correct_decoding_percentage[1:28:4],
    #                  'bo') #B
    #     axx1[0].plot(p_val_correct[2:28:4], correct_decoding_percentage[2:28:4],
    #                  'ro') #A
    #     axx1[0].plot(p_val_correct[3:28:4], correct_decoding_percentage[3:28:4],
    #                  'bo') #B
    #
    #     axx1[1].plot(p_val_edge[0:28:4], edge_decoding_percentage[0:28:4],
    #                  'ro')
    #     axx1[1].plot(p_val_edge[1:28:4], edge_decoding_percentage[1:28:4],
    #                  'bo')
    #     axx1[1].plot(p_val_edge[2:28:4], edge_decoding_percentage[2:28:4],
    #                  'ro')
    #     axx1[1].plot(p_val_edge[3:28:4], edge_decoding_percentage[3:28:4],
    #                  'bo')
    #
    # axx1[0].set_ylim((-0.1, 1.1))
    # axx1[0].set_xlim((-0.1, 1.1))
    # axx1[0].set_title('P value Vs. correct decoding fraction', fontsize=18)
    # axx1[0].set_ylabel('correct decoding fraction', fontsize=16)
    # axx1[0].set_xlabel('P value', fontsize=16)
    # axx1[1].set_ylim((-0.1, 1.1))
    # axx1[1].set_xlim((-0.1, 1.1))
    # axx1[1].set_title('P value Vs. edge decoding fraction', fontsize=18)
    # axx1[1].set_ylabel('edge decoding fraction', fontsize=16)
    # axx1[1].set_xlabel('P value', fontsize=16)
    #
    # for i in range(2):
    #     for xtick, ytick in zip(axx1[i].xaxis.get_major_ticks(),
    #                             axx1[i].yaxis.get_major_ticks()):
    #         xtick.label.set_fontsize(15)
    #         ytick.label.set_fontsize(15)
    #
    #
    # f1.show()

    ####### Plot decoding histogram and number of events per bin decoding ######

    for i, mouse in enumerate(MOUSE):
        f2, axx2 = subplots(2, 2, sharex=True, sharey='row')
        npzfile = np.load('bucket_decoding_results_c%sm%s.npz'\
                          %(CAGE[i], mouse))
        decoded_bins_all_sessions = npzfile['decoded_bins_all_sessions']
        decoded_env_all_sessions = npzfile['decoded_env_all_sessions']
        number_of_events_per_frame_all_sessions = npzfile \
        ['number_of_events_per_frame_all_sessions']

        decoded_bins_all_sessions = np.concatenate(decoded_bins_all_sessions)
        decoded_env_all_sessions = np.concatenate(decoded_env_all_sessions)
        number_of_events_per_frame_all_sessions = np.concatenate(
            number_of_events_per_frame_all_sessions)

        env_A_indices = decoded_env_all_sessions == 0
        env_B_indices = decoded_env_all_sessions == 1

        box_data_A = divide_to_boxes\
            (number_of_events_per_frame_all_sessions[env_A_indices],
             decoded_bins_all_sessions[env_A_indices])

        box_data_B = divide_to_boxes\
             (number_of_events_per_frame_all_sessions[env_B_indices],
             decoded_bins_all_sessions[env_B_indices])

        axx2[0, 0].boxplot(box_data_A)
        axx2[0, 0].set_xlabel('Decoded bin')
        axx2[0, 0].set_ylabel('Number of events in frame')
        axx2[0, 0].set_title('Linear track')
        axx2[0, 1].boxplot(box_data_B)
        axx2[0, 1].set_xlabel('Decoded bin')
        axx2[0, 1].set_ylabel('Number of events in frame')
        axx2[0, 1].set_title('L-shape track')
        axx2[1, 0].hist(decoded_bins_all_sessions[env_A_indices], normed=True)
        axx2[1, 0].set_xlabel('Decoded bin')
        axx2[1, 0].set_ylabel('Probability of decoding')
        axx2[1, 1].hist(decoded_bins_all_sessions[env_B_indices], normed=True)
        axx2[1, 1].set_xlabel('Decoded bin')
        axx2[1, 1].set_ylabel('Probability of decoding')
        f2.suptitle('Bucket decoding C%sM%s' % (CAGE[i], mouse))
        f2.show()

    raw_input('press enter')

if __name__ == '__main__':
    main()