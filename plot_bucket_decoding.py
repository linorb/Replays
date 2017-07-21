import numpy as np
from matplotlib.pyplot import *
from decode_bucket_trials import MOUSE, CAGE

f1, axx1 = subplots(1, 2, sharey=True, sharex=True)
def main():
    for i, mouse in enumerate(MOUSE):
        npzfile = np.load('bucket_decoding_statistics_c%sm%s.npz'\
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
    raw_input('press enter')

if __name__ == '__main__':
    main()