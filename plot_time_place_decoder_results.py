import numpy as np
from matplotlib.pyplot import *

MOUSE = [4, 4, 1, 1]
CAGE = [6, 7, 11, 13]
mouse_color = [(1, 0.5, 0),
               (0.5, 0, 0.5),
               (0, 0.5, 0.5),
               (0.5, 0.5, 0)]

def main():
    MAE_all_miceA = []
    chance_level_all_miceA = []
    MAE_all_miceB = []
    chance_level_all_miceB = []
    xticklabel = []

    f, axx = subplots(1, 2, sharey=True)
    mouse_plot = []
    for i in range(len(MOUSE)):
        npzfile = np.load(r'results\linear_track_decoding_results_c%dm%d.npz'\
                          %(CAGE[i], MOUSE[i]))
        # The zero is the decoding only for the trials from the same day
        MAE_all_miceA.append(npzfile['mean_error_all_sessions'][0])
        MAE_permutation = npzfile['mean_error_permutaion_all_sessions'][0]
        chance_level_per_trial = np.array([np.mean(x) for x in MAE_permutation])
        chance_level_all_miceA.append(chance_level_per_trial)

        npzfile = np.load(r'results\Lshape_track_decoding_results_c%sm%s.npz' \
                          % (CAGE[i], MOUSE[i]))
        MAE_all_miceB.append(npzfile['mean_error_all_sessions'][0])
        MAE_permutation = npzfile['mean_error_permutaion_all_sessions'][0]
        chance_level_per_trial = np.array([np.mean(x) for x in MAE_permutation])
        chance_level_all_miceB.append(chance_level_per_trial)

        xticklabel.append('C%dM%d' %(CAGE[i], MOUSE[i]))

    # # put the legend outside the axis. taken from:
    # #  https://matplotlib.org/users/legend_guide.html
    # legend(bbox_to_anchor=(1.1, 1.05))
    # axx[0].axhline(0, color='black')
    # axx[0].axvline(0, color='black')
    # axx[0].grid()

    meam_MAEA = np.array([np.mean(x) for x in MAE_all_miceA])
    std_MAEA = np.array([np.std(x) for x in MAE_all_miceA])
    meam_chanceA = np.array([np.mean(x) for x in chance_level_all_miceA])
    std_chanceA = np.array([np.std(x) for x in chance_level_all_miceA])
    meam_MAEB = np.array([np.mean(x) for x in MAE_all_miceB])
    std_MAEB = np.array([np.std(x) for x in MAE_all_miceB])
    meam_chanceB = np.array([np.mean(x) for x in chance_level_all_miceB])
    std_chanceB = np.array([np.std(x) for x in chance_level_all_miceB])

    f, axx = subplots(1, 2, sharey=True, sharex=True)
    axx[0].errorbar(np.arange(1,5,1), meam_MAEA, yerr=std_MAEA, color='red', fmt='o')
    axx[0].errorbar(np.arange(1,5,1),meam_chanceA, yerr=std_chanceA, color='gray', fmt='o')
    setp(axx[0], xticks=np.arange(1, 5, 1), xticklabels=xticklabel)
    axx[0].set_ylabel('Absolut median error', fontsize=25)
    axx[0].set_title('Environment A', fontsize=25)
    line1 = axx[1].errorbar(np.arange(1,5,1),meam_MAEB, yerr=std_MAEB,
                             color='red',fmt='o', label='Performance')
    line2 = axx[1].errorbar(np.arange(1,5,1),meam_chanceB, yerr=std_chanceB,
                             color='gray',fmt='o', label='Chance level')
    legend(bbox_to_anchor=(1.1, 1.), fontsize=20)
    setp(axx[1], xticks=np.arange(1,5,1), xticklabels = xticklabel)
    axx[1].set_title('Environment B', fontsize=25)
    axx[1].set_xlim(0, 5)

    for i in range(2):
        for xtick in axx[i].xaxis.get_major_ticks():
           xtick.label.set_fontsize(22)
        for ytick in axx[i].yaxis.get_major_ticks():
            ytick.label.set_fontsize(22)


    f.show()
    raw_input('press enter  ')
if __name__ == '__main__':
    main()