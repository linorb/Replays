import numpy as np
import matplotlib.pyplot as plt

def main():
##  Plot probability of the linear track
    f, axx = plt.subplots(2, 2, sharex='row', sharey=True)
    ######### For linear track #########:

    ### Load the data ###
    filename = r'results\Linear_edge_probability.npz'
    npzfile = np.load(filename)
    p_edge_run = npzfile['p_edge_run'].all()
    p_edge_no_run = npzfile['p_edge_no_run'].all()

    MOUSE = [3, 6, 6, 3, 0, 4, 4, 1, 1]
    CAGE = [40, 40, 38, 38, 38, 6, 7, 11, 13]
    mouse_color = [(1, 0, 0),  # red
                   (0, 1, 0),  # green
                   (0, 0, 1),  # blue
                   (1, 1, 0),  #
                   (0, 1, 1),  # cyan
                   (1, 0, 1),  # purple
                   (1, 0.5, 0),
                   (0.5, 0, 0.5),
                   (0, 0.5, 0.5),
                   (0.5, 0.5, 0)]

    # Unite data from all mice
    p_edge_run_all_mice = []
    p_edge_no_run_all_mice = []
    number_of_sessions = 0
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        number_of_sessions += len(p_edge_run[mouse_name])
        p_edge_run_all_mice.append(np.concatenate(p_edge_run[mouse_name]))
        p_edge_no_run_all_mice.append(np.concatenate(p_edge_no_run[mouse_name]))

    print 'Number of session environment A:' , number_of_sessions
    p_edge_run_all_mice = np.concatenate(p_edge_run_all_mice)
    p_edge_no_run_all_mice = np.concatenate(p_edge_no_run_all_mice)

    axx[0, 0].hist(p_edge_run_all_mice[~np.isnan(p_edge_run_all_mice)],
                   normed=True)
    axx[0, 0].set_title('P(active in edge|active in run)', fontsize=18)
    axx[0, 0].set_ylabel('Environment A \n \n Density', fontsize=18)
    axx[0, 1].hist(p_edge_no_run_all_mice[~np.isnan(p_edge_no_run_all_mice)],
                   normed=True)
    axx[0, 1].set_title('P(active in edge|not active in run)', fontsize=18)

    ######### For L-shape track #########:

    ### Load the data ###
    filename = r'results\Lshape_edge_probability.npz'
    npzfile = np.load(filename)
    p_edge_run = npzfile['p_edge_run'].all()
    p_edge_no_run = npzfile['p_edge_no_run'].all()

    MOUSE = [4, 4, 1, 1]
    CAGE = [6, 7, 11, 13]
    mouse_color = [(1, 0.5, 0),
                   (0.5, 0, 0.5),
                   (0, 0.5, 0.5),
                   (0.5, 0.5, 0)]

    # Unite data from all mice
    p_edge_run_all_mice = []
    p_edge_no_run_all_mice = []
    number_of_sessions = 0
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        number_of_sessions += len(p_edge_run[mouse_name])
        p_edge_run_all_mice.append(np.concatenate(p_edge_run[mouse_name]))
        p_edge_no_run_all_mice.append(np.concatenate(p_edge_no_run[mouse_name]))

    print 'Number of session environment B:', number_of_sessions
    p_edge_run_all_mice = np.concatenate(p_edge_run_all_mice)
    p_edge_no_run_all_mice = np.concatenate(p_edge_no_run_all_mice)

    axx[1, 0].hist(p_edge_run_all_mice[~np.isnan(p_edge_run_all_mice)],
                   normed=True)
    axx[1, 0].set_ylabel('Environment B \n \n Density', fontsize=18)
    axx[1, 0].set_xlabel('Probability for activation', fontsize=18)
    axx[1, 1].hist(p_edge_no_run_all_mice[~np.isnan(p_edge_no_run_all_mice)],
                   normed=True)
    axx[1, 1].set_xlabel('Probability for activation', fontsize=18)
    # plt.yscale('log', nonposy='clip')
    for i in range(2):
        for j in range(2):
            for xtick in axx[i, j].xaxis.get_major_ticks():
                xtick.label.set_fontsize(15)
            for ytick in axx[i, j].yaxis.get_major_ticks():
                ytick.label.set_fontsize(15)

    f.show()
    raw_input('press enter')

if __name__ == '__main__':
    main()