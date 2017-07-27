import numpy as np
from matplotlib.pyplot import *

MOUSE = [4, 4, 1, 1, 6, 3, 6, 3, 0]
CAGE = [6, 7, 11, 13, 40, 40, 38, 38, 38]

def main():
    MAE_all_miceA = []
    xticklabelA = []
    MAE_all_miceB = []
    xticklabelB = []
    for i in range(4):
        npzfile = np.load(r'results\linear_track_decoding_results_c%dm%d.npz'\
                          %(CAGE[i], MOUSE[i]))
        MAE_all_miceA.append(npzfile.f.arr_0[0])
        xticklabelA.append('C%dM%d' %(CAGE[i], MOUSE[i]))

    for i in range(4):
        npzfile = np.load(r'results\Lshape_track_decoding_results_c%sm%s.npz' \
                          % (CAGE[i], MOUSE[i]))
        MAE_all_miceB.append(npzfile.f.arr_0[0])
        xticklabelB.append('C%dM%d' % (CAGE[i], MOUSE[i]))

    f, axx = subplots(1, 2, sharey=True)
    axx[0].boxplot(MAE_all_miceA)
    setp(axx[0], xticks=np.arange(1, 5, 1), xticklabels=xticklabelA)
    axx[0].set_ylabel('Absolut median error', fontsize=15)
    axx[0].set_title('Environment A')
    axx[1].boxplot(MAE_all_miceB)
    setp(axx[1], xticks=np.arange(1,5,1), xticklabels = xticklabelB)
    axx[1].set_title('Environment B')
    f.suptitle('Maximum-likelihood decoder performance per mouse', fontsize=18)

    for i in range(2):
        for xtick in axx[i].xaxis.get_major_ticks():
           xtick.label.set_fontsize(15)
        for ytick in axx[i].yaxis.get_major_ticks():
            ytick.label.set_fontsize(15)


    f.show()
    raw_input('press enter  ')
if __name__ == '__main__':
    main()